#include <memory>
#include <optional>
#include "binder/bound_order_by.h"
#include "catalog/catalog.h"
#include "catalog/column.h"
#include "concurrency/transaction.h"
#include "execution/expressions/array_expression.h"
#include "execution/expressions/column_value_expression.h"
#include "execution/expressions/comparison_expression.h"
#include "execution/expressions/constant_value_expression.h"
#include "execution/expressions/vector_expression.h"
#include "execution/plans/abstract_plan.h"
#include "execution/plans/filter_plan.h"
#include "execution/plans/index_scan_plan.h"
#include "execution/plans/limit_plan.h"
#include "execution/plans/projection_plan.h"
#include "execution/plans/seq_scan_plan.h"
#include "execution/plans/sort_plan.h"
#include "execution/plans/topn_plan.h"
#include "execution/plans/vector_index_scan_plan.h"
#include "fmt/core.h"
#include "optimizer/optimizer.h"
#include "type/type.h"
#include "type/type_id.h"

namespace bustub {

auto MatchVectorIndex(const Catalog &catalog, table_oid_t table_oid, uint32_t col_idx, VectorExpressionType dist_fn,
                      const std::string &vector_index_match_method) -> const IndexInfo * {
  // IMPLEMENT ME
  const auto &index_info_vec = catalog.GetTableIndexes(catalog.GetTable(table_oid)->name_);
  for (auto index_info : index_info_vec) {
    auto vec_index = dynamic_cast<VectorIndex *>(index_info->index_.get());
    if (vec_index != nullptr) {
      // std::cout << "method:" << vector_index_match_method << "|||-------------------" << vec_index->GetName()
      //           << std::endl;
      // std::cout << "size:" << vec_index->GetKeyAttrs().size() << " " << vec_index->GetKeyAttrs()[0] << " " << col_idx
      //           << std::endl;
      if (vec_index->GetKeyAttrs().size() == 1 && vec_index->GetKeyAttrs()[0] == col_idx &&
          vec_index->distance_fn_ == dist_fn) {
        if (vector_index_match_method == "") {
          return index_info;
        }
        if ((vector_index_match_method == "ivfflat" && index_info->index_type_ == IndexType::VectorIVFFlatIndex) ||
            (vector_index_match_method == "hnsw" && index_info->index_type_ == IndexType::VectorHNSWIndex)) {
          return index_info;
        }
      }
    }
  }
  // not found
  return nullptr;
}

auto Optimizer::OptimizeAsVectorIndexScan(const AbstractPlanNodeRef &plan) -> AbstractPlanNodeRef {
  std::vector<AbstractPlanNodeRef> children;
  for (const auto &child : plan->GetChildren()) {
    children.emplace_back(OptimizeAsVectorIndexScan(child));
  }
  auto optimized_plan = plan->CloneWithChildren(std::move(children));
  // IMPLEMENT ME
  if (optimized_plan->GetType() == PlanType::TopN && optimized_plan->GetChildAt(0)->GetType() == PlanType::Projection &&
      optimized_plan->GetChildAt(0)->GetChildAt(0)->GetType() == PlanType::SeqScan) {
    const auto &topN_plan = dynamic_cast<const TopNPlanNode &>(*optimized_plan);
    const auto &project_plan = dynamic_cast<const ProjectionPlanNode &>(*optimized_plan->GetChildAt(0));
    const auto &seq_scan_plan = dynamic_cast<const SeqScanPlanNode &>(*optimized_plan->GetChildAt(0)->GetChildAt(0));
    size_t limit = topN_plan.GetN();
    BUSTUB_ASSERT(topN_plan.GetOrderBy().size() == 1 && (topN_plan.GetOrderBy()[0].first == OrderByType::ASC ||
                                                         topN_plan.GetOrderBy()[0].first == OrderByType::DEFAULT),
                  "not exepected");
    auto compare = topN_plan.GetOrderBy()[0].second;
    auto vector_express = dynamic_cast<VectorExpression *>(compare.get());
    if (vector_express != nullptr) {
      auto left_expr = vector_express->children_[0];
      auto right_expr = vector_express->children_[1];
      auto col_expr = dynamic_cast<ColumnValueExpression *>(right_expr.get());
      auto array_expr = dynamic_cast<ArrayExpression *>(left_expr.get());
      auto col_index = col_expr->GetColIdx();
      // sort ----> project  col index
      // project ---->origin col index(real_col)
      auto real_col_expr = dynamic_cast<ColumnValueExpression *>(project_plan.expressions_[col_index].get());
      BUSTUB_ASSERT(real_col_expr != nullptr, "not expected case");
      col_index = real_col_expr->GetColIdx();
      if (auto index = MatchVectorIndex(catalog_, seq_scan_plan.table_oid_, col_index, vector_express->expr_type_,
                                        vector_index_match_method_);
          index != nullptr && col_expr != nullptr && array_expr != nullptr) {
        auto base_vector = std::make_shared<const ArrayExpression>(*array_expr);
        auto vector_index_plan = std::make_shared<VectorIndexScanPlanNode>(
            seq_scan_plan.output_schema_, seq_scan_plan.table_oid_, seq_scan_plan.table_name_, index->index_oid_,
            index->index_->GetName(), base_vector, limit);
        return std::make_shared<ProjectionPlanNode>(project_plan.output_schema_, project_plan.expressions_,
                                                    vector_index_plan);
      }
    }
  }
  // may not good
  if (optimized_plan->GetType() == PlanType::TopN && optimized_plan->GetChildAt(0)->GetType() == PlanType::SeqScan) {
    BUSTUB_ASSERT(optimized_plan->children_.size() == 1, "must exactly one child");
    const auto &topN_plan = dynamic_cast<const TopNPlanNode &>(*optimized_plan);
    const auto &seq_scan_plan = dynamic_cast<const SeqScanPlanNode &>(*optimized_plan->GetChildAt(0));
    size_t limit = topN_plan.GetN();
    BUSTUB_ASSERT(topN_plan.GetOrderBy().size() == 1 && (topN_plan.GetOrderBy()[0].first == OrderByType::ASC ||
                                                         topN_plan.GetOrderBy()[0].first == OrderByType::DEFAULT),
                  "not exepected");
    auto compare = topN_plan.GetOrderBy()[0].second;
    auto vector_express = dynamic_cast<VectorExpression *>(compare.get());
    if (vector_express != nullptr) {
      auto left_expr = vector_express->children_[0];
      auto right_expr = vector_express->children_[1];
      auto col_expr = dynamic_cast<ColumnValueExpression *>(right_expr.get());
      auto array_expr = dynamic_cast<ArrayExpression *>(left_expr.get());
      auto col_index = col_expr->GetColIdx();
      if (auto index = MatchVectorIndex(catalog_, seq_scan_plan.table_oid_, col_index, vector_express->expr_type_,
                                        vector_index_match_method_);
          index != nullptr && col_expr != nullptr && array_expr != nullptr) {
        auto base_vector = std::make_shared<const ArrayExpression>(*array_expr);
        return std::make_shared<VectorIndexScanPlanNode>(plan->output_schema_, seq_scan_plan.table_oid_,
                                                         seq_scan_plan.table_name_, index->index_oid_,
                                                         index->index_->GetName(), base_vector, limit);
      }
    }
  }
  return optimized_plan;
}

}  // namespace bustub
