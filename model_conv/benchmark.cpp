#include <iostream>

#include <models/tree.h>
#include <models/plt.h>

#include <ctime>
#include <set>
#include <filesystem>

#include <pecos/core/xmc/inference.hpp>
#include <pecos/core/utils/scipy_loader.hpp>

pecos::csr_t OnlyFirstRow(pecos::csr_t mat) {
	pecos::csr_t out;

	auto nnz = mat.indptr[1];

	out.allocate(1, mat.cols, nnz);

	std::memcpy(out.indices, mat.indices, nnz * sizeof(pecos::csr_t::index_type));
	std::memcpy(out.val, mat.val, nnz * sizeof(pecos::csr_t::value_type));
	std::memcpy(out.indptr, mat.indptr, 2 * sizeof(pecos::csr_t::mem_index_type));

	return out;
}

SRMatrix<Feature> PecosToNapkinXC(const pecos::csr_t& mat, double bias) {
	SRMatrix<Feature> m;
	
	for (int row = 0; row < mat.rows; ++row) {

		auto start = mat.indptr[row];
		auto end = mat.indptr[row+1];
		auto len = end - start;

		std::vector<Feature> row_features;
		row_features.reserve(len);

		for (auto i = start; i < end; ++i) {
			Feature f;
			f.index = mat.indices[i];
			f.value = mat.val[i];
			row_features.emplace_back(f);
		}

		if (bias > 0.0) {
			Feature f;
			f.index = mat.cols;
			f.value = bias;
			row_features.emplace_back(f);
		}

		m.appendRow(row_features);
	}

	return m;
}

std::vector<std::vector<Prediction>> PecosPredictionToNapkinXC(const pecos::csr_t& mat) {
	std::vector<std::vector<Prediction>> result;
	result.reserve(mat.rows);
	
	for (int row = 0; row < mat.rows; ++row) {

		auto start = mat.indptr[row];
		auto end = mat.indptr[row+1];
		auto len = end - start;

		std::vector<Prediction> row_vec;
		row_vec.reserve(len);

		for (auto i = start; i < end; ++i) {
			Prediction p;
			p.label = mat.indices[i];
			p.value = mat.val[i];
			row_vec.emplace_back(p);
		}

		result.emplace_back(std::move(row_vec));
	}

	return result;
}

void ComputeRecallPrecision(
	const std::vector<std::vector<Prediction>>& ground_truth, 
	const std::vector<std::vector<Prediction>>& predictions,
	int topK,
	std::vector<double>& recall,
	std::vector<double>& precision) {

	recall.resize(topK);
	precision.resize(topK);

	std::fill(recall.begin(), recall.end(), 0.0);
	std::fill(precision.begin(), precision.end(), 0.0);

	for (int i = 0; i < ground_truth.size(); ++i) {
		auto& truth = ground_truth[i];
		auto& prediction = predictions[i];

		std::set<int> truth_labels;

		for (auto& t : truth) {
			truth_labels.emplace(t.label);
		}

		for (int k = 1; k <= topK; ++k) {
			std::set<int> pred_labels;
			std::set<int> pred_truth_labels;

			for (int j = 0; j < k; ++j) {
				pred_labels.emplace(prediction[j].label);
			}

			std::set_intersection(truth_labels.begin(), truth_labels.end(), 
				pred_labels.begin(), pred_labels.end(),
				std::inserter(pred_truth_labels, pred_truth_labels.begin()));

			recall[k-1] += (double)pred_truth_labels.size() / (double)truth_labels.size();
			precision[k-1] += (double)pred_truth_labels.size() / (double)pred_labels.size();
		}
	}

	for (auto& r : recall) {
		r /= (double)ground_truth.size();
	}

	for (auto& p : precision) {
		p /= (double)ground_truth.size();
	}
}

void TestDataSet(const std::filesystem::path& path) {

	// Verify that we have both a napkin and pecos model
	auto pecos_path = path / "model";
	auto napkin_path = path / "napkin-model";

	if (!std::filesystem::exists(pecos_path) || !std::filesystem::is_directory(pecos_path)) {
		std::cout << path << " does not have a PECOS model. Skipping..." << std::endl;
		return;
	}

	if (!std::filesystem::exists(napkin_path) || !std::filesystem::is_directory(napkin_path)) {
		std::cout << path << " does not have a Napkin-XC model. Skipping..." << std::endl;
		return;
	}

	pecos::csr_t X;
	pecos::csr_t Y;

	std::vector<std::vector<Prediction>> pecos_predictions;
	std::vector<std::vector<Prediction>> napkin_predictions;

	int top_k = 10;
	int beam_size = 20; 

	{
		std::cout << "Loading " << path / "X.tst.tfidf.npz" << "..." << std::endl;
		pecos::ScipyCsrF32Npz X_npz(path / "X.tst.tfidf.npz");
		std::cout << "Loading " << path / "Y.tst.npz" << "..." << std::endl;
		pecos::ScipyCsrF32Npz Y_npz(path / "Y.tst.npz");
		X = pecos::csr_npz_to_csr_t_deep_copy(X_npz);
		Y = pecos::csr_npz_to_csr_t_deep_copy(Y_npz);
	}

	{
		std::cout << "Loading PECOS model " << pecos_path << "..." << std::endl;
		pecos::HierarchicalMLModel model(pecos_path, pecos::LAYER_TYPE_HASH_CHUNKED);

		std::cout << "Running PECOS Prediction..." << std::endl;
		pecos::csr_t Y_pred;

		std::clock_t c_start = std::clock();
		model.predict<pecos::csr_t, pecos::csr_t>(X, Y_pred, beam_size, "sigmoid", top_k, 1);
		std::clock_t c_end = std::clock();

		long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
		std::cout << "CPU time per query: " << time_elapsed_ms / (double)X.rows << " ms\n";

		pecos_predictions = PecosPredictionToNapkinXC(Y_pred);
		Y_pred.free_underlying_memory();

		std::cout << std::endl;
	}

	{
		std::cout << "Loading NapkinXC model " << napkin_path << "..." << std::endl;
		auto current_dir = std::filesystem::current_path();

		std::filesystem::current_path(napkin_path);

		Args args;
		args.loadFromFile("args.bin");

		BatchPLT model_;
		model_.load(args, args.output);

		SRMatrix<Feature> X_f = PecosToNapkinXC(X, 1.0);

		std::cout << "Running NapkinXC Prediction..." << std::endl;

		args.topK = top_k;
		args.beamSearchWidth = beam_size;
		args.threads = 1;
		args.treeSearchType = TreeSearchType::beam;

		std::clock_t c_start = std::clock();
		napkin_predictions = model_.predictBatch(X_f, args);
		std::clock_t c_end = std::clock();

		for (auto& pred : napkin_predictions) {
			pred.resize(std::min<int>(pred.size(), args.topK));
		}

		long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
		std::cout << "CPU time per query: " << time_elapsed_ms  / (double)X.rows << " ms\n";

		std::filesystem::current_path(current_dir);

		std::cout << std::endl;
	}

	std::vector<double> pecos_recall;
	std::vector<double> pecos_precision;

	std::vector<double> napkin_recall;
	std::vector<double> napkin_precision;

	auto truth = PecosPredictionToNapkinXC(Y);

	ComputeRecallPrecision(truth, pecos_predictions, top_k, pecos_recall, pecos_precision);
	ComputeRecallPrecision(truth, napkin_predictions, top_k, napkin_recall, napkin_precision);

	auto printPrecisionRecall = [top_k](const std::string& header, 
		const std::vector<double>& precision,
		const std::vector<double>& recall) {
		std::cout << "=========== " << header << " =============" << std::endl;
		std::cout << std::setw(10) << "prec@k";
		for (int i = 0; i < top_k; ++i) {
			std::cout << std::setw(10) << precision[i];
		}
		std::cout << std::endl;
		std::cout << std::setw(10) << "recall@k";
		for (int i = 0; i < top_k; ++i) {
			std::cout << std::setw(10) << recall[i];
		}
		std::cout << std::endl << std::endl;
	};

	printPrecisionRecall("PECOS", pecos_precision, pecos_recall);
	printPrecisionRecall("NapkinXC", napkin_precision, napkin_recall);

	X.free_underlying_memory();
	Y.free_underlying_memory();
}

int main(int argc, char *argv[]) {

	std::vector<std::filesystem::path> data_dirs;

	if (argc < 2) {
		auto path = std::filesystem::path(DATA_DIR);

		for (auto entry : std::filesystem::directory_iterator(path)) {
			if (entry.is_directory()) {
				data_dirs.emplace_back(entry.path());
			}
		}
	} else {
		for (int i = 1; i < argc; ++i) {
			data_dirs.emplace_back(argv[i]);
		}
	}

	for (auto dir : data_dirs) {
		if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
			TestDataSet(dir);
		}
	}
}