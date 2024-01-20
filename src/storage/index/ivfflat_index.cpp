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

struct HashFunc {
  size_t operator()(const Vector &vec) const {
    size_t ans = 0;
    for (double v : vec) {
      ans += v;
    }
    return ans;
  }
};
struct HashCmp {
  bool operator()(const Vector &a, const Vector &b) const {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  }
};
typedef std::unordered_map<Vector, Vector, HashFunc, HashCmp> Vector_Vector_Map;
typedef std::unordered_map<Vector, bool, HashFunc, HashCmp> Vector_Bool_Map;
typedef std::unordered_map<Vector, bool, HashFunc, HashCmp> Vector_Double_Map;
typedef std::unordered_map<Vector, std::unordered_map<Vector, double, HashFunc, HashCmp>, HashFunc, HashCmp> VectorMap;
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
    BUSTUB_ASSERT(!vec.empty(), "must not empty");
    Vector origin_vec(vec[0].first.size(), 0);
    for (size_t j = 0; j < vec.size(); j++) {
      VectorAdd(origin_vec, vec[j].first);
    }
    VectorScalarDiv(origin_vec, centroids_.size());
    centroids_.push_back(origin_vec);
  }

  VectorMap l_map;
  VectorMap d_map;  // x c
  Vector_Double_Map u_map;
  Vector_Vector_Map c_map;
  Vector_Bool_Map r_map;
  for (auto data : initial_data) {
    for (auto center : centroids_) {
      l_map[data.first][center] = 0;
    }
  }
  for (auto data : initial_data) {
    double cur_max = DBL_MAX;
    Vector vec;
    for (auto center : centroids_) {
      // compute
      double dis = ComputeDistance(data.first, center, distance_fn_);
      if (cur_max > dis) {
        vec = center;
        cur_max = dis;
      }
      l_map[data.first][center] = dis;
    }
    u_map[data.first] = cur_max;
    c_map[data.first] = vec;
  }
  // for repeat
  for (int i = 0; i < 500; i++) {
    VectorMap d_center_map;
    Vector_Double_Map s_map;
    Vector_Bool_Map r_map;
    Vector_Vector_Map m_map;
    for (auto data : initial_data) {
      r_map[data.first] = true;
    }
    // step 1
    for (auto center1 : centroids_) {
      double max_ans = DBL_MAX;
      for (auto center2 : centroids_) {
        if (!HashCmp()(center1, center2)) {
          double dis = ComputeDistance(center1, center2, distance_fn_);
          max_ans = std::min(dis, max_ans);
          d_center_map[center1][center2] = dis;
        }
      }
      s_map[center1] = max_ans / 2;
    }
    // step 2~3
    for (auto data : initial_data) {
      if (u_map[data.first] > s_map[c_map[data.first]]) continue;
      auto x = data.first;
      for (auto center : centroids_) {
        if (!HashCmp()(center, c_map[x]) &&
            u_map[data.first] > std::max(l_map[x][center], d_center_map[center][c_map[x]] / 2)) {
          // 3.a
          double d_x_c;
          if (r_map[x]) {
            r_map[x] = false;
            d_x_c = ComputeDistance(x, c_map[x], distance_fn_);
          } else {
            d_x_c = u_map[x];
          }
          // 3.b
          if (d_x_c > l_map[x][center] || d_x_c > d_center_map[center][c_map[x]] / 2) {
            if (ComputeDistance(x, center, distance_fn_) < d_x_c) {
              c_map[x] = center;
            }
          }
        }
      }
    }
    // 4. compute m(c)
    std::unordered_map<Vector, std::vector<Vector>, HashFunc, HashCmp> aggregate_map;
    for (auto data : initial_data) {
      aggregate_map[c_map[data.first]].push_back(data.first);
    }
    for (auto &[center, vec] : aggregate_map) {
      BUSTUB_ASSERT(vec.size() >= 1, "vec.size is 0");
      Vector ans(vec[0].size(), 0);
      for (auto v : vec) {
        VectorAdd(ans, v);
      }
      VectorScalarDiv(ans, vec.size());
      m_map[center] = ans;
    }
    // init
    for (auto center : centroids_) {
      if (!m_map.count(center)) {
        d_center_map[center][center] = 0;
        continue;
      }
      d_center_map[center][m_map[center]] = ComputeDistance(center, m_map[center], distance_fn_);
    }
    // 5
    for (auto data : initial_data) {
      for (auto center : centroids_) {
        l_map[data.first][center] = std::max(0.0, l_map[data.first][center] - d_center_map[center][m_map[center]]);
      }
    }
    // 6
    for (auto data : initial_data) {
      u_map[data.first] = u_map[data.first] + d_center_map[c_map[data.first]][m_map[c_map[data.first]]];
      r_map[data.first] = true;
    }
    // ( implement extra part): can optizmar use index stanfor the center
    for (auto data : initial_data) {
      auto vec = c_map[data.first];
      if (m_map.count(vec)) {
        c_map[data.first] = m_map[vec];
      }
    }
    // 7
    centroids_.clear();
    for (auto [center_orgin, center] : m_map) {
      if (center.size() == 0) {
        centroids_.push_back(center_orgin);
        continue;
      }
      centroids_.push_back(center);
    }
    BUSTUB_ASSERT(centroids_.size() == lists_, "not match size");
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
