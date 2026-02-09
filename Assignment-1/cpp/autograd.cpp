#include "autograd.hpp"
#include <unordered_set>
#include <stdexcept> // <--- ADDED THIS

// Helper to handle shared_ptr in set
static void topo_dfs(
    TensorPtr node,
    std::vector<TensorPtr>& graph,
    std::unordered_set<Tensor*>& visited // Use raw ptr for set key
) {
    if (visited.count(node.get())) return;
    visited.insert(node.get());

    for (const auto& parent : node->parents) {
        topo_dfs(parent, graph, visited);
    }

    graph.push_back(node);
}

void topo_sort(TensorPtr node, std::vector<TensorPtr>& graph) {
    std::unordered_set<Tensor*> visited;
    topo_dfs(node, graph, visited);
}

void backward(TensorPtr loss) {
    if (loss->size() != 1) {
        throw std::runtime_error("backward() expects scalar loss tensor");
    }

    if (loss->grad.empty()) {
        loss->grad.resize(1, 0.0f);
    }
    loss->grad[0] = 1.0f;

    std::vector<TensorPtr> graph;
    topo_sort(loss, graph);

    for (auto it = graph.rbegin(); it != graph.rend(); ++it) {
        TensorPtr t = *it;
        if (t->backward_fn) {
            t->backward_fn();
        }
    }
}