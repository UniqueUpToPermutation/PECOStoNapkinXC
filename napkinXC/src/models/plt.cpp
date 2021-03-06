/*
 Copyright (c) 2018-2021 by Marek Wydmuch, Kalina Jasinska-Kobus, Robert Istvan Busa-Fekete

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <list>
#include <vector>

#include "plt.h"


PLT::PLT() {
    tree = nullptr;
    treeSize = 0;
    treeDepth = 0;
    nodeEvaluationCount = 0;
    nodeUpdateCount = 0;
    dataPointCount = 0;
    type = plt;
    name = "PLT";
}

void PLT::unload() {
    for (auto b : bases) delete b;
    bases.clear();
    bases.shrink_to_fit();
    delete tree;
}

void PLT::assignDataPoints(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                           std::vector<std::vector<double>>& binWeights, SRMatrix<Label>& labels,
                           SRMatrix<Feature>& features, Args& args) {

    Log(CERR) << "Assigning data points to nodes ...\n";

    // Positive and negative nodes
    UnorderedSet<TreeNode*> nPositive;
    UnorderedSet<TreeNode*> nNegative;

    // Gather examples for each node
    int rows = features.rows();
    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        nPositive.clear();
        nNegative.clear();

        getNodesToUpdate(nPositive, nNegative, labels[r], labels.size(r));
        addNodesLabelsAndFeatures(binLabels, binFeatures, nPositive, nNegative, features[r]);

        nodeUpdateCount += nPositive.size() + nNegative.size();
        ++dataPointCount;
    }

    unsigned long long usedMem = nodeUpdateCount * (sizeof(double) + sizeof(Feature*)) + binLabels.size() * (sizeof(binLabels) + sizeof(binFeatures));
    Log(CERR) << "  Temporary data size: " << formatMem(usedMem) << "\n";
}

void PLT::getNodesToUpdate(UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative,
                           const int* rLabels, const int rSize) {
    for (int i = 0; i < rSize; ++i) {
        auto ni = tree->leaves.find(rLabels[i]);
        if (ni == tree->leaves.end()) {
            Log(CERR) << "Encountered example with label " << rLabels[i] << " that does not exists in the tree\n";
            continue;
        }
        TreeNode* n = ni->second;
        nPositive.insert(n);
        while (n->parent) {
            n = n->parent;
            nPositive.insert(n);
        }
    }

    if (nPositive.empty()) {
        nNegative.insert(tree->root);
        return;
    }

    for(auto& n : nPositive) {
        for (const auto &child : n->children) {
            if (!nPositive.count(child))
                nNegative.insert(child);
        }
    }
}

void PLT::addNodesLabelsAndFeatures(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                      UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative,
                      Feature* features) {
    for (const auto& n : nPositive) {
        binLabels[n->index].push_back(1.0);
        binFeatures[n->index].push_back(features);
    }

    for (const auto& n : nNegative) {
        binLabels[n->index].push_back(0.0);
        binFeatures[n->index].push_back(features);
    }
}

std::vector<std::vector<Prediction>> PLT::predictBatch(SRMatrix<Feature>& features, Args& args) {
    if (args.treeSearchType == exact) return Model::predictBatch(features, args);
    else if (args.treeSearchType == beam) return predictWithBeamSearch(features, args);
    else throw std::invalid_argument("Unknown tree search type");
}

std::vector<std::vector<Prediction>> PLT::predictWithBeamSearch(SRMatrix<Feature>& features, Args& args){
    Log(CERR) << "Starting prediction in 1 thread ...\n";

    int rows = features.rows();
    int nodes = tree->nodes.size();

    std::vector<std::vector<Prediction>> prediction(rows);
    std::vector<std::vector<TreeNodeValue>> levelPredictions(rows);
    std::vector<std::vector<Prediction>> nodePredictions(nodes);

    auto nextLevelQueue = new std::queue<TreeNode*>();
    nextLevelQueue->push(tree->root);
    for(int i = 0; i < rows; ++i) nodePredictions[tree->root->index].emplace_back(i, 1.0);

    int nCount = 0;
    while(!nextLevelQueue->empty()){
        printProgress(nCount++, nodes);

        auto levelQueue = nextLevelQueue;
        nextLevelQueue = new std::queue<TreeNode*>();

        // Predict for level
        while(!levelQueue->empty()){
            auto n = levelQueue->front();
            levelQueue->pop();
            int nIdx = n->index;

            //std::cerr << "\nnIdx: " << nIdx << "\n";

            if(!nodePredictions[nIdx].empty()){
                //std::cerr << "from initial base\n";
                auto base = bases[nIdx];
                auto type = base->getType();
                if(type == sparse) 
					base->to(dense);

                for(auto &e : nodePredictions[nIdx]){
                    int rIdx = e.label;
                    double prob = bases[nIdx]->predictProbability(features[rIdx]) * e.value;
                    double value = prob;

                    // Reweight score
                    if (!labelsWeights.empty()) value *= nodesWeights[nIdx].weight;

                    if(n->label >= 0) 
						prediction[rIdx].emplace_back(n->label, value); // Final prediction
                    else levelPredictions[rIdx].emplace_back(n, prob, value); // Internal node prediction
                }
                nodeEvaluationCount += nodePredictions[nIdx].size();
                nodePredictions[nIdx].clear();

                //std::cerr << "back to initial base\n";
                if(type != base->getType()) 
					base->to(type);
            }

            for(auto &c : n->children)
                nextLevelQueue->push(c);
        }
        delete levelQueue;

        // Keep top predictions and prepare next level
        for(int rIdx = 0; rIdx < rows; ++rIdx){
            auto &v = levelPredictions[rIdx];

            if(!thresholds.empty()){
                int j = 0;
                for(int i = 0; i < v.size(); ++i){
                    if(v[i].value > nodesThr[v[i].node->index].th)
                        v[j++] = v[i];
                }
                v.resize(j - 1);
            }
            else {
                std::sort(v.rbegin(), v.rend());

                if(args.threshold > 0){
                    int i = 0;
                    while (i < v.size() && v[i++].value > args.threshold);
                    v.resize(i - 1);
                }
                else v.resize(std::min(v.size(), (size_t)args.beamSearchWidth));
            }

            for(auto &nv : v)
                for(auto &c : nv.node->children)
                    nodePredictions[c->index].emplace_back(rIdx, nv.prob);
            v.clear();
        }
    }
    delete nextLevelQueue;

    for(int rIdx = 0; rIdx < rows; ++rIdx){
        auto &v = prediction[rIdx];
        std::sort(v.rbegin(), v.rend());
    }

    dataPointCount = rows;
    return prediction;
}

void PLT::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    int topK = args.topK;
    double threshold = args.threshold;

    if(topK > 0) prediction.reserve(topK);
    TopKQueue<TreeNodeValue> nQueue(args.topK);


    // Set functions
    std::function<bool(TreeNode*, double)> ifAddToQueue = [&] (TreeNode* node, double prob) {
        return true;
    };

    if(args.threshold > 0)
        ifAddToQueue = [&] (TreeNode* node, double prob) {
            return (prob >= threshold);
        };
    else if(thresholds.size())
        ifAddToQueue = [&] (TreeNode* node, double prob) {
            return (prob >= nodesThr[node->index].th);
        };

    std::function<double(TreeNode*, double)> calculateValue = [&] (TreeNode* node, double prob) {
        return prob;
    };

    if (!labelsWeights.empty())
        calculateValue = [&] (TreeNode* node, double prob) {
            return prob * nodesWeights[node->index].weight;
        };

    // Predict for root
    double rootProb = predictForNode(tree->root, features);
    addToQueue(ifAddToQueue, calculateValue, nQueue, tree->root, rootProb);
    ++nodeEvaluationCount;
    ++dataPointCount;

    Prediction p = predictNextLabel(ifAddToQueue, calculateValue, nQueue, features);
    while ((prediction.size() < topK || topK == 0) && p.label != -1) {
        prediction.push_back(p);
        p = predictNextLabel(ifAddToQueue, calculateValue, nQueue, features);
    }
}

Prediction PLT::predictNextLabel(
    std::function<bool(TreeNode*, double)>& ifAddToQueue, std::function<double(TreeNode*, double)>& calculateValue,
    TopKQueue<TreeNodeValue>& nQueue, Feature* features) {
    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top();
        nQueue.pop();

        if (!nVal.node->children.empty()) {
            for (const auto& child : nVal.node->children)
                addToQueue(ifAddToQueue, calculateValue, nQueue, child, nVal.prob * predictForNode(child, features));
            nodeEvaluationCount += nVal.node->children.size();
        }
        if (nVal.node->label >= 0) return {nVal.node->label, nVal.value};
    }

    return {-1, 0};
}

void PLT::calculateNodesLabels(){
    if(tree->t != nodesLabels.size()){
        nodesLabels.clear();
        nodesLabels.resize(tree->t);

        for (auto& l : tree->leaves) {
            TreeNode* n = l.second;
            while (n != nullptr) {
                nodesLabels[n->index].push_back(l.first);
                n = n->parent;
            }
        }
    }
}

void PLT::setNodeThreshold(TreeNode* n){
    TreeNodeThrExt& nTh = nodesThr[n->index];
    nTh.th = 1;
    for (auto &l : nodesLabels[n->index]) {
        if (thresholds[l] < nTh.th) {
            nTh.th = thresholds[l];
            nTh.label = l;
        }
    }
}

void PLT::setNodeWeight(TreeNode* n){
    TreeNodeWeightsExt& nW = nodesWeights[n->index];
    nW.weight = 0;
    for (auto &l : nodesLabels[n->index]) {
        if (labelsWeights[l] > nW.weight) {
            nW.weight = labelsWeights[l];
            nW.label = l;
        }
    }
}

void PLT::setThresholds(std::vector<double> th){
    Model::setThresholds(th);
    if(tree) {
        calculateNodesLabels();
        if (tree->t != nodesThr.size()) nodesThr.resize(tree->t);
        for (auto& n : tree->nodes) setNodeThreshold(n);
    }
}

void PLT::setLabelsWeights(std::vector<double> lw){
    Model::setLabelsWeights(lw);
    if(tree) {
        calculateNodesLabels();
        if (tree->t != nodesWeights.size()) nodesWeights.resize(tree->t);
        for (auto& n : tree->nodes) setNodeWeight(n);
    }
}

void PLT::updateThresholds(UnorderedMap<int, double> thToUpdate){
    for(auto& th : thToUpdate)
        thresholds[th.first] = th.second;

    for(auto& th : thToUpdate){
        TreeNode* n = tree->leaves[th.first];
        TreeNodeThrExt& nTh = nodesThr[n->index];
        while(n != tree->root){
            if(th.second < nTh.th){
                nTh.th = th.second;
                nTh.label = th.first;
            } else if (th.first == nTh.label && th.second > nTh.th){
                setNodeThreshold(n);
            }
            n = n->parent;
        }
    }
}

double PLT::predictForLabel(Label label, Feature* features, Args& args) {
    auto fn = tree->leaves.find(label);
    if(fn == tree->leaves.end()) return 0;
    TreeNode* n = fn->second;
    double value = bases[n->index]->predictProbability(features);
    while (n->parent) {
        n = n->parent;
        value *= predictForNode(n, features);
        ++nodeEvaluationCount;
    }

    if(!labelsWeights.empty())
        value = labelsWeights[label] = value;

    return value;
}

void PLT::load(Args& args, std::string infile) {
    Log(CERR) << "Loading " << name << " model ...\n";

    tree = new Tree();
    tree->loadFromFile(joinPath(infile, "tree.bin"));
    bases = loadBases(joinPath(infile, "weights.bin"), args.resume, args.loadAs);

    assert(bases.size() == tree->nodes.size());
    m = tree->getNumberOfLeaves();

    loaded = true;
}

void PLT::printInfo() {
    Log(COUT) << name << " additional stats:"
              << "\n  Tree size: " << (tree != nullptr ? tree->nodes.size() : treeSize)
              << "\n  Tree depth: " << (tree != nullptr ? tree->getTreeDepth() : treeDepth) << "\n";
    if(nodeUpdateCount > 0)
        Log(COUT) << "  Updated estimators / data point: " << static_cast<double>(nodeUpdateCount) / dataPointCount << "\n";
    if(nodeEvaluationCount > 0)
        Log(COUT) << "  Evaluated estimators / data point: " << static_cast<double>(nodeEvaluationCount) / dataPointCount << "\n";
}

void PLT::buildTree(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output){
    delete tree;
    tree = new Tree();
    tree->buildTreeStructure(labels, features, args);
    m = tree->getNumberOfLeaves();

    // Save tree and free it, it is no longer needed
    tree->saveToFile(joinPath(output, "tree.bin"));
    tree->saveTreeStructure(joinPath(output, "tree"));
    treeSize = tree->nodes.size();
    treeDepth = tree->getTreeDepth();
    assert(treeSize == tree->t);
}

std::vector<std::vector<std::pair<int, double>>> PLT::getNodesToUpdate(std::vector<std::vector<Label>>& labels){
    if(!tree) throw std::runtime_error("Tree is not constructed, build a tree first");

    // Positive and negative nodes
    UnorderedSet<TreeNode*> nPositive;
    UnorderedSet<TreeNode*> nNegative;

    Log(CERR) << "Getting nodes to update ...\n";

    // Gather examples for each node
    int rows = labels.size();
    std::vector<std::vector<std::pair<int, double>>> nodesToUpdate(rows);

    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        nPositive.clear();
        nNegative.clear();

        getNodesToUpdate(nPositive, nNegative, labels[r].data(), labels[r].size());
        nodesToUpdate[r].reserve(nPositive.size() + nNegative.size());
        for (const auto& n : nPositive) nodesToUpdate[r].emplace_back(n->index, 1.0);
        for (const auto& n : nNegative) nodesToUpdate[r].emplace_back(n->index, 0);
    }

    return nodesToUpdate;
}

std::vector<std::vector<std::pair<int, double>>> PLT::getNodesUpdates(std::vector<std::vector<Label>>& labels){
    if(!tree) throw std::runtime_error("Tree is not constructed, build a tree first");

    // Positive and negative nodes
    UnorderedSet<TreeNode*> nPositive;
    UnorderedSet<TreeNode*> nNegative;

    Log(CERR) << "Getting nodes to update ...\n";

    // Gather examples for each node
    int rows = labels.size();
    std::vector<std::vector<std::pair<int, double>>> nodesDataPoints(m);

    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        nPositive.clear();
        nNegative.clear();

        getNodesToUpdate(nPositive, nNegative, labels[r].data(), labels[r].size());
        for (const auto& n : nPositive) nodesDataPoints[n->index].emplace_back(r, 1.0);
        for (const auto& n : nNegative) nodesDataPoints[n->index].emplace_back(r, 0);
    }

    return nodesDataPoints;
}

void BatchPLT::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) {
    if(!tree) buildTree(labels, features, args, output);

    Log(CERR) << "Training tree ...\n";

    // Check data
    //assert(tree->k >= labels.cols());

    // Examples selected for each node
    std::vector<std::vector<double>> binLabels(tree->t);
    std::vector<std::vector<Feature*>> binFeatures(tree->t);
    std::vector<std::vector<double>> binWeights;

    if (type == hsm && args.pickOneLabelWeighting) binWeights.resize(tree->t);
    else binWeights.emplace_back(features.rows(), 1);

    assignDataPoints(binLabels, binFeatures, binWeights, labels, features, args);

    // Train bases
    std::vector<ProblemData> binProblemData;
    if (type == hsm && args.pickOneLabelWeighting)
        for(int i = 0; i < treeSize; ++i) binProblemData.emplace_back(binLabels[i], binFeatures[i], features.cols(), binWeights[i]);
    else
        for (int i = 0; i < treeSize; ++i) binProblemData.emplace_back(binLabels[i], binFeatures[i], features.cols(), binWeights[0]);

    for (auto &pb: binProblemData) {
        pb.r = features.rows();
        pb.invPs = 1;
    }
    
    // Delete tree, it is no longer needed
    delete tree;
    tree = nullptr;
    
    trainBases(joinPath(output, "weights.bin"), binProblemData, args);
}
