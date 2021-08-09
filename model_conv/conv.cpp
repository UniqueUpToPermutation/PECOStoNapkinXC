#include <iostream>

#include <models/tree.h>
#include <models/plt.h>

#include <filesystem>

#include <pecos/core/xmc/inference.hpp>

struct NodeClassRanges {
	int _start;
	int _end;
};

// Build the tree
Tree* makeTreeFrom(pecos::HierarchicalMLModel& model) {
	Tree* tree = new Tree();
	TreeNode* root = tree->createTreeNode();
	tree->root = root;

	int base_index = 0;

	root->index = base_index++;
	root->parent = nullptr;

	std::vector<TreeNode*> lastLayer = { root };
	std::vector<TreeNode*> nextLayer;

	for (auto layer : model.get_model_layers()) {
		auto layer_cast = dynamic_cast<pecos::MLModel<pecos::csc_t>*>(layer);
		auto& layer_data = layer_cast->get_layer_data();

		auto nextLayerSize = layer_data.C.rows;
		for (int i = 0; i < nextLayerSize; ++i) {
			auto node = tree->createTreeNode();
			node->index = base_index++;
			nextLayer.emplace_back(node);
		}

		for (int parent = 0; parent < lastLayer.size(); ++parent) {
			auto start = layer_data.C.indptr[parent];
			auto end = layer_data.C.indptr[parent+1];
			auto parentNode = lastLayer[parent];

			for (auto idx = start; idx < end; ++idx) {
				auto child_idx = layer_data.C.indices[idx];
				auto childNode = nextLayer[child_idx];

				tree->setParent(childNode, parentNode);
			}
		}

		lastLayer = std::move(nextLayer);
		nextLayer.clear();
	}

	int current_label = 0;
	for (auto node : lastLayer) {
		tree->setLabel(node, current_label++);
	}

	for (int nodeId = tree->nodes.size() - 1; nodeId >= 0; --nodeId) {
		auto node = tree->nodes[nodeId];
		auto& children = node->children;

		if (children.size() == 0) {
			node->subtreeLeaves = 1;

		} else {
			int subtreeLeaves = 0;
			for (auto& child : children) {
				subtreeLeaves += child->subtreeLeaves;
			}
			node->subtreeLeaves = subtreeLeaves;
		}
	}

	tree->k = tree->leaves.size();
	tree->t = tree->nodes.size();

	return tree;
}

// Make the weight vectors
std::vector<Base*> makeBasesFrom(pecos::HierarchicalMLModel& model) {
	std::vector<Base*> result;

	Base* b = new Base();
	b->W = nullptr;
	b->firstClass = 0;
	b->classCount = 0;
	b->lossType = LossType::logistic;

	result.emplace_back(b);

	for (auto layer : model.get_model_layers()) {
		auto layer_cast = dynamic_cast<pecos::MLModel<pecos::csc_t>*>(layer);
		auto& layer_data = layer_cast->get_layer_data();
		auto& weights = layer_data.W;

		for (int i = 0; i < weights.cols; ++i) {

			auto start = weights.indptr[i];
			auto end = weights.indptr[i+1];
			auto len = end - start;

			auto vec = new MapVector<Weight>(len);

			for (auto idx = start; idx < end; ++idx) {
				vec->insertD(weights.indices[idx], weights.data[idx]);
			}

			Base* b = new Base();
			b->W = vec;
			b->firstClass = 1; // WTF?
			b->classCount = 2; // WTF?
			b->lossType = LossType::logistic;

			result.emplace_back(b);
		}
	}

	return result;
}

void ConvertModel(std::filesystem::path path, std::filesystem::path out_path) {
	std::filesystem::path _model_dir_in = path;
	std::filesystem::path _model_dir_out = out_path;

	{
		std::cout << "Loading PECOS model from " << _model_dir_in << "..." << std::endl;

		pecos::HierarchicalMLModel model(_model_dir_in, pecos::LAYER_TYPE_CSC);

		Tree* tree = makeTreeFrom(model);
		std::vector<Base*> bases = makeBasesFrom(model);

		if (!std::filesystem::exists(_model_dir_out)) {
			std::filesystem::create_directory(_model_dir_out);
		} 
		
		std::ofstream _os_bases;
		std::ofstream _os_tree;
		std::ofstream _os_args;
		
		if (std::filesystem::is_directory(_model_dir_out)) {

			std::filesystem::path _bases_out = _model_dir_out / "weights.bin";
			std::filesystem::path _tree_out = _model_dir_out / "tree.bin";
			std::filesystem::path _args_out = _model_dir_out / "args.bin";

			_os_bases = std::ofstream(_bases_out);
			_os_tree = std::ofstream(_tree_out);
			_os_args = std::ofstream(_args_out);

		} else {
			throw std::runtime_error("Output directory exists and is not a directory!");
		}

		std::cout << "Saving NapkinXC model to " << _model_dir_out << "..." << std::endl;

		int size = bases.size();
		_os_bases.write((char*)&size, sizeof(size));
		for (int i = 0; i < bases.size(); ++i) {
			Base* base = bases[i];
			base->save(_os_bases, false);
			delete base;
		}
		_os_bases.close();

		tree->save(_os_tree);
		delete tree;
		_os_tree.close();

		Args args;
		args.modelType = ModelType::plt;

		args.save(_os_args);
		_os_args.close();
	}

	{
		std::cout << "Verifying that NapkinXC can load model..." << std::endl;

		auto current_dir = std::filesystem::current_path();

		std::filesystem::current_path(_model_dir_out);
		Args args;
		args.loadFromFile("args.bin");		

       	BatchPLT model_;
		model_.load(args, args.output);

		std::filesystem::current_path(current_dir);

		std::cout << "Verification successful!" << std::endl;	
	}
}

int main(int argc, char *argv[]) {

	std::vector<std::filesystem::path> model_dirs;

	if (argc < 2) {
		std::filesystem::path data_dir = DATA_DIR;
		model_dirs.emplace_back(data_dir / "eurlex-4k" / "model");
		model_dirs.emplace_back(data_dir / "amazoncat-13k" / "model");
		model_dirs.emplace_back(data_dir / "wiki10-31k" / "model");
		model_dirs.emplace_back(data_dir / "wiki-500k" / "model");
		model_dirs.emplace_back(data_dir / "amazon-670k" / "model");
	} else {
		for (int i = 1; i < argc; ++i) {
			model_dirs.emplace_back(argv[i]);
		}
	}

	for (auto dir_entry : model_dirs) {
		std::filesystem::path pecosModelPath = dir_entry;
		std::filesystem::path napkinXCModelPath = dir_entry / ".." / "napkin-model";
		if (std::filesystem::exists(pecosModelPath)) {
			std::cout << "Found PECOS model " << pecosModelPath << "..." << std::endl;
			ConvertModel(pecosModelPath, napkinXCModelPath);
		}
	}
}