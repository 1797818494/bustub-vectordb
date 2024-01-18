//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// insert_executor.cpp
//
// Identification: src/execution/insert_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "execution/executors/insert_executor.h"

namespace bustub {

InsertExecutor::InsertExecutor(ExecutorContext *exec_ctx, const InsertPlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(std::move(child_executor)) {}

void InsertExecutor::Init() {
  // init child
  child_executor_->Init();
}

auto InsertExecutor::Next([[maybe_unused]] Tuple *tuple, RID *rid) -> bool {
  // 已经返回一个tuple
  if (emitted_) {
    return false;
  }
  TableInfo *table_info = exec_ctx_->GetCatalog()->GetTable(plan_->GetTableOid());
  Tuple insert_tuple;
  RID insert_rid;
  int insert_count = 0;
  while (child_executor_->Next(&insert_tuple, &insert_rid)) {
    std::optional<RID> result_rid = table_info->table_->InsertTuple(TupleMeta{0, false}, insert_tuple);
    std::vector<bustub::IndexInfo *> index_vec = exec_ctx_->GetCatalog()->GetTableIndexes(table_info->name_);
    for (auto index_info : index_vec) {
      VectorIndex *vec_index = dynamic_cast<VectorIndex *>(index_info->index_.get());
      if (nullptr != vec_index) {
        auto key_vec_one = index_info->index_->GetKeyAttrs();
        BUSTUB_ASSERT(key_vec_one.size() == 1, "vector index not only has one vector type");
        BUSTUB_ASSERT(result_rid.has_value(), "insert vector fail");
        std::vector<double> insert_vec =
            insert_tuple.GetValue(&child_executor_->GetOutputSchema(), key_vec_one[0]).GetVector();
        vec_index->InsertVectorEntry(insert_vec, result_rid.value());
      }
    }
    insert_count++;
  }
  emitted_ = true;
  std::vector<Value> vals;
  vals.push_back(Value(TypeId::INTEGER, insert_count));
  count_tuple_ = Tuple(vals, &plan_->OutputSchema());
  *tuple = count_tuple_;
  return true;
}

}  // namespace bustub
