#include "execution/executors/sort_executor.h"

namespace bustub {

SortExecutor::SortExecutor(ExecutorContext *exec_ctx, const SortPlanNode *plan,
                           std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(std::move(child_executor)) {}

void SortExecutor::Init() {
  child_executor_->Init();
  sort_tuples_.clear();
  Tuple child_tuple;
  RID child_rid;
  while (child_executor_->Next(&child_tuple, &child_rid)) {
    sort_tuples_.push_back(child_tuple);
  };
  const auto &order_vec = plan_->GetOrderBy();
  sort(sort_tuples_.begin(), sort_tuples_.end(), [=](const Tuple &a, const Tuple b) {
    for (auto order_predicate : order_vec) {
      BUSTUB_ASSERT(order_predicate.first != OrderByType::INVALID, "invalid order type");
      auto val_a = order_predicate.second->Evaluate(&a, plan_->OutputSchema());
      auto val_b = order_predicate.second->Evaluate(&b, plan_->OutputSchema());
      if (val_a.CompareEquals(val_b) == GetCmpBool(true)) {
        continue;
      }
      return (OrderByType::ASC == order_predicate.first || OrderByType::DEFAULT == order_predicate.first)
                 ? (val_a.CompareLessThan(val_b) == GetCmpBool(true))
                 : !(val_a.CompareLessThan(val_b) == GetCmpBool(true));
    }
    return true;
  });
  pos_ = 0;
}
auto SortExecutor::Next(Tuple *tuple, RID *rid) -> bool {
  if (pos_ == sort_tuples_.size()) {
    return false;
  }
  *tuple = sort_tuples_[pos_];
  *rid = sort_tuples_[pos_].GetRid();
  pos_++;
  return true;
}

}  // namespace bustub
