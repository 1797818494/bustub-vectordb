//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// seq_scan_executor.cpp
//
// Identification: src/execution/seq_scan_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/seq_scan_executor.h"

namespace bustub {

SeqScanExecutor::SeqScanExecutor(ExecutorContext *exec_ctx, const SeqScanPlanNode *plan)
    : AbstractExecutor(exec_ctx), plan_(plan) {}

void SeqScanExecutor::Init() {
  TableHeap *table_heap = exec_ctx_->GetCatalog()->GetTable(plan_->GetTableOid())->table_.get();
  iter_ = std::make_unique<TableIterator>(table_heap, table_heap->MakeIterator().GetRID(),
                                          table_heap->MakeIterator().GetLastRID());
}

auto SeqScanExecutor::Next(Tuple *tuple, RID *rid) -> bool {
  if (iter_->IsEnd()) {
    return false;
  }
  *tuple = iter_->GetTuple().second;
  *rid = iter_->GetRID();
  ++(*iter_);
  return true;
}

}  // namespace bustub
