#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/macros.h"
#include "execution/expressions/vector_expression.h"
#include "fmt/format.h"
#include "fmt/std.h"
#include "storage/index/hnsw_index.h"
#include "storage/index/index.h"
#include "storage/index/vector_index.h"

namespace bustub {
HNSWIndex::HNSWIndex(std::unique_ptr<IndexMetadata> &&metadata, BufferPoolManager *buffer_pool_manager,
                     VectorExpressionType distance_fn, const std::vector<std::pair<std::string, int>> &options)
    : VectorIndex(std::move(metadata), distance_fn),
      vertices_(std::make_unique<std::vector<Vector>>()),
      layers_{{*vertices_, distance_fn}} {
  std::optional<size_t> m;
  std::optional<size_t> ef_construction;
  std::optional<size_t> ef_search;
  for (const auto &[k, v] : options) {
    if (k == "m") {
      m = v;
    } else if (k == "ef_construction") {
      ef_construction = v;
    } else if (k == "ef_search") {
      ef_search = v;
    }
  }
  if (!m.has_value() || !ef_construction.has_value() || !ef_search.has_value()) {
    throw Exception("missing options: m / ef_construction / ef_search for hnsw index");
  }
  ef_construction_ = *ef_construction;
  m_ = *m;
  ef_search_ = *ef_search;
  m_max_ = m_;
  m_max_0_ = m_ * m_;
  layers_[0].m_max_ = m_max_0_;
  m_l_ = 1.0 / std::log(m_);
  std::random_device rand_dev;
  generator_ = std::mt19937(rand_dev());
}
typedef std::pair<std::vector<double>, size_t> PVS;
auto SelectNeighbors(const std::vector<double> &vec, const std::vector<size_t> &vertex_ids,
                     const std::vector<std::vector<double>> &vertices, size_t m, VectorExpressionType dist_fn)
    -> std::vector<size_t> {
  return {};
}

auto NSW::SearchLayer(const std::vector<double> &base_vector, size_t limit, const std::vector<size_t> &entry_points)
    -> std::vector<size_t> {
  auto greater_distance = [this, base_vector](const PVS &vec_a, const PVS &vec_b) {
    return ComputeDistance(vec_a.first, base_vector, dist_fn_) > ComputeDistance(vec_b.first, base_vector, dist_fn_);
  };
  auto less_distance = [this, base_vector](const PVS &vec_a, const PVS &vec_b) {
    return ComputeDistance(vec_a.first, base_vector, dist_fn_) < ComputeDistance(vec_b.first, base_vector, dist_fn_);
  };
  std::unordered_set<size_t> visited;
  std::priority_queue<PVS, std::vector<PVS>, decltype(greater_distance)> min_heap(greater_distance);  // C
  std::priority_queue<PVS, std::vector<PVS>, decltype(less_distance)> max_heap(less_distance);        // W
  // entry -> C / W
  for (auto id : entry_points) {
    min_heap.push({vertices_[id], id});
    max_heap.push({vertices_[id], id});
    visited.insert(id);
  }
  while (!min_heap.empty()) {
    auto node = min_heap.top();
    min_heap.pop();
    if (ComputeDistance(base_vector, node.first, dist_fn_) >
        ComputeDistance(base_vector, max_heap.top().first, dist_fn_)) {
      break;
    }
    for (auto neighbor : edges_[node.second]) {
      if (!visited.count(neighbor)) {
        visited.insert(neighbor);
        max_heap.push({vertices_[neighbor], neighbor});
        min_heap.push({vertices_[neighbor], neighbor});
        // retain knn
        while (max_heap.size() > limit) {
          max_heap.pop();
        }
      }
    }
  }
  while (!min_heap.empty()) {
    min_heap.pop();
  }
  while (!max_heap.empty()) {
    min_heap.push(max_heap.top());
    max_heap.pop();
  }
  std::vector<size_t> result_rids;
  for (size_t i = 0; i < limit && !min_heap.empty(); i++) {
    result_rids.push_back(min_heap.top().second);
    min_heap.pop();
  }
  return result_rids;
}

auto NSW::AddVertex(size_t vertex_id) { in_vertices_.push_back(vertex_id); }

auto NSW::Insert(const std::vector<double> &vec, size_t vertex_id, size_t ef_construction, size_t m) {
  // IMPLEMENT ME
  // AddVertex(vertex_id);
  auto vec_pos = SearchLayer(vec, ef_construction, std::vector<size_t>(1, DefaultEntryPoint()));
  std::vector<size_t> reach_max_pos;
  for (auto vertex : vec_pos) {
    Connect(vertex, vertex_id);
    if (edges_[vertex].size() == m_max_) {
      reach_max_pos.push_back(vertex);
    }
  }
  for (auto pos : reach_max_pos) {
    // recompute the KNN and cut oters
    auto knn_vec = SearchLayer(vertices_[pos], ef_construction, std::vector<size_t>(1, DefaultEntryPoint()));
    for (auto target : edges_[pos]) {
      if (std::find_if(knn_vec.begin(), knn_vec.end(), [target](size_t num) { return target == num; }) ==
          knn_vec.end()) {
        std::remove(edges_[target].begin(), edges_[target].end(), pos);
        std::remove(edges_[pos].begin(), edges_[pos].end(), target);
      }
    }
  }
}

void NSW::Connect(size_t vertex_a, size_t vertex_b) {
  edges_[vertex_a].push_back(vertex_b);
  edges_[vertex_b].push_back(vertex_a);
}

auto HNSWIndex::AddVertex(const std::vector<double> &vec, RID rid) -> size_t {
  auto id = vertices_->size();
  vertices_->emplace_back(vec);
  rids_.emplace_back(rid);
  return id;
}

void HNSWIndex::BuildIndex(std::vector<std::pair<std::vector<double>, RID>> initial_data) {
  std::shuffle(initial_data.begin(), initial_data.end(), generator_);

  for (const auto &[vec, rid] : initial_data) {
    InsertVectorEntry(vec, rid);
  }
}

auto HNSWIndex::ScanVectorKey(const std::vector<double> &base_vector, size_t limit) -> std::vector<RID> {
  size_t max_level = layers_.size() - 1;
  std::vector<size_t> eps(1, layers_.back().DefaultEntryPoint());

  for (int i = max_level; i >= 0; i--) {
    eps = layers_[i].SearchLayer(base_vector, 1, eps);
  }
  auto vertex_ids = layers_[0].SearchLayer(base_vector, limit, eps);
  std::vector<RID> result;
  result.reserve(vertex_ids.size());
  for (const auto &id : vertex_ids) {
    result.push_back(rids_[id]);
  }
  return result;
}
auto HNSWIndex::ComputeLevel() -> size_t {
  // 生成一个 [0,1) 范围内的随机数
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  double random_number = unif(generator_);

  // 计算 level 值
  double level = -std::log(random_number) * m_l_;

  // 向下取整
  size_t floor_level = std::floor(level);

  return floor_level;
}

void HNSWIndex::InsertVectorEntry(const std::vector<double> &key, RID rid) {
  size_t random_level = ComputeLevel();
  size_t max_level = layers_.size() - 1;
  auto id = AddVertex(key, rid);
  if (random_level > max_level) {
    for (size_t i = max_level + 1; i <= random_level; i++) {
      layers_.push_back(NSW{*vertices_, distance_fn_});  // use edges to identify the existing node
      layers_[i].AddVertex(id);
      layers_[i].Insert(key, id, ef_construction_, m_);
    }
  }
  for (size_t i = 0; i <= std::min(random_level, max_level); i++) {
    layers_[i].AddVertex(id);
  }
  std::cerr << "random level: " << random_level << " " << max_level << " " << layers_.size() << std::endl;
  std::vector<size_t> eps(1, layers_.back().DefaultEntryPoint());
  for (size_t i = layers_.size() - 1; i >= random_level + 1; i--) {
    eps = layers_[i].SearchLayer(key, 1, eps);
  }
  for (int i = random_level; i >= 0; i--) {
    eps = layers_[i].SearchLayer(key, ef_construction_, eps);
    layers_[i].Insert(key, id, ef_construction_, m_);
  }
  // layers_[0].Insert(key, id, ef_construction_, m_);
}
}  // namespace bustub
