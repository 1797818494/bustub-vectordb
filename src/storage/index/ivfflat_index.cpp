#include "storage/index/ivfflat_index.h"
#include <algorithm>
#include <optional>
#include <random>
#include "common/exception.h"
#include "execution/expressions/vector_expression.h"
#include "storage/index/index.h"
#include "storage/index/vector_index.h"

namespace bustub {
using Vector = std::vector<double>;

IVFFlatIndex::IVFFlatIndex(std::unique_ptr<IndexMetadata> &&metadata, BufferPoolManager *buffer_pool_manager,
                           VectorExpressionType distance_fn, const std::vector<std::pair<std::string, int>> &options)
    : VectorIndex(std::move(metadata), distance_fn) {
  std::optional<size_t> lists;
  std::optional<size_t> probe_lists;
  for (const auto &[k, v] : options) {
    if (k == "lists") {
      lists = v;
    } else if (k == "probe_lists") {
      probe_lists = v;
    }
  }
  if (!lists.has_value() || !probe_lists.has_value()) {
    throw Exception("missing options: lists / probe_lists for ivfflat index");
  }
  lists_ = *lists;
  probe_lists_ = *probe_lists;
}

void VectorAdd(Vector &a, const Vector &b) {
  for (size_t i = 0; i < a.size(); i++) {
    a[i] += b[i];
  }
}

void VectorScalarDiv(Vector &a, double x) {
  for (auto &y : a) {
    y /= x;
  }
}

// Find the nearest centroid to the base vector in all centroids
auto FindCentroid(const Vector &vec, const std::vector<Vector> &centroids, VectorExpressionType dist_fn) -> size_t {
  double ans = DBL_MAX;
  size_t index = -1;
  for (size_t i = 0; i < centroids.size(); i++) {
    double compute_res = ComputeDistance(vec, centroids[i], dist_fn);
    if (ans > compute_res) {
      index = i;
      ans = compute_res;
    }
  }
  return index;
}

// Compute new centroids based on the original centroids.
auto FindCentroids(const std::vector<std::pair<Vector, RID>> &data, const std::vector<Vector> &centroids,
                   VectorExpressionType dist_fn) -> std::vector<Vector> {
  std::vector<std::vector<std::pair<Vector, RID>>> buckets(centroids.size());
  std::vector<Vector> new_centroids;
  for (auto vert : data) {
    auto centroid = FindCentroid(vert.first, centroids, dist_fn);
    buckets[centroid].push_back(vert);
  }
  for (size_t i = 0; i < buckets.size(); i++) {
    const auto &vec = buckets[i];
    if (vec.empty()) continue;
    Vector origin_vec(vec[0].first.size(), 0);
    for (size_t j = 0; j < vec.size(); j++) {
      VectorAdd(origin_vec, vec[j].first);
    }
    VectorScalarDiv(origin_vec, centroids.size());
    new_centroids.push_back(origin_vec);
  }
  return new_centroids;
}

void IVFFlatIndex::BuildIndex(std::vector<std::pair<Vector, RID>> initial_data) {
  if (initial_data.empty()) {
    return;
  }
  // random init
  std::vector<std::vector<std::pair<Vector, RID>>> buckets(lists_);
  for (auto vert : initial_data) {
    size_t index = rand() % lists_;
    buckets[index].push_back(vert);
  }
  for (size_t i = 0; i < buckets.size(); i++) {
    const auto &vec = buckets[i];
    if (vec.empty()) continue;
    Vector origin_vec(vec[0].first.size(), 0);
    for (size_t j = 0; j < vec.size(); j++) {
      VectorAdd(origin_vec, vec[j].first);
    }
    VectorScalarDiv(origin_vec, centroids_.size());
    centroids_.push_back(origin_vec);
  }
  for (size_t i = 0; i < 500; i++) {
    centroids_ = FindCentroids(initial_data, centroids_, distance_fn_);
  }

  // insert may be after the build
  centroids_buckets_.resize(lists_);
  for (auto &[vec, rid] : initial_data) {
    InsertVectorEntry(vec, rid);
  }
}

void IVFFlatIndex::InsertVectorEntry(const std::vector<double> &key, RID rid) {
  size_t index = FindCentroid(key, centroids_, distance_fn_);
  centroids_buckets_[index].push_back({key, rid});
}

auto IVFFlatIndex::ScanVectorKey(const std::vector<double> &base_vector, size_t limit) -> std::vector<RID> {
  auto greater_distance_pos = [base_vector, this](const std::pair<Vector, size_t> &v1,
                                                  const std::pair<Vector, size_t> &v2) -> bool {
    return ComputeDistance(base_vector, v1.first, distance_fn_) > ComputeDistance(base_vector, v2.first, distance_fn_);
  };
  auto greater_distance_pair = [base_vector, this](const std::pair<Vector, RID> &p1,
                                                   const std::pair<Vector, RID> &p2) -> bool {
    return ComputeDistance(base_vector, p1.first, distance_fn_) > ComputeDistance(base_vector, p2.first, distance_fn_);
  };
  std::priority_queue<std::pair<Vector, size_t>, std::vector<std::pair<Vector, size_t>>, decltype(greater_distance_pos)>
      centroids_probe_queue(greater_distance_pos);
  std::priority_queue<std::pair<Vector, RID>, std::vector<std::pair<Vector, RID>>, decltype(greater_distance_pair)>
      knn_queue(greater_distance_pair);
  // can optimzer
  for (size_t i = 0; i < centroids_.size(); i++) {
    centroids_probe_queue.push({centroids_[i], i});
  }

  for (size_t i = 0; i < probe_lists_ && !centroids_probe_queue.empty(); i++) {
    auto pair = centroids_probe_queue.top();
    centroids_probe_queue.pop();
    for (auto &pair_vec_rid : centroids_buckets_[pair.second]) {
      knn_queue.push(pair_vec_rid);
    }
  }
  std::vector<RID> rids;
  for (size_t i = 0; i < limit && !knn_queue.empty(); i++) {
    auto pair = knn_queue.top();
    knn_queue.pop();
    rids.push_back(pair.second);
  }
  BUSTUB_ASSERT(rids.size() == limit, "size not match");
  return rids;
}

}  // namespace bustub
