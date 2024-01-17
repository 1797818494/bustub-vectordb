#include "execution/executors/topn_executor.h"

namespace bustub {

TopNExecutor::TopNExecutor(ExecutorContext *exec_ctx, const TopNPlanNode *plan,
                           std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(std::move(child_executor)) {}

void TopNExecutor::Init() {
  child_executor_->Init();
  auto order_vec = plan_->GetOrderBy();
  auto less = [this](const Tuple &a, const Tuple b) {
    for (auto order_predicate : this->plan_->order_bys_) {
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
  };
  std::priority_queue<Tuple, std::vector<Tuple>, decltype(less)> max_heap(less);
  Tuple child_tuple;
  RID child_rid;
  while (child_executor_->Next(&child_tuple, &child_rid)) {
    if (max_heap.size() == plan_->n_) {
      max_heap.push(child_tuple);
      max_heap.pop();
    } else {
      max_heap.push(child_tuple);
    }
  }
  while (!max_heap.empty()) {
    emit_tuples_.push(max_heap.top());
    max_heap.pop();
  }
}

auto TopNExecutor::Next(Tuple *tuple, RID *rid) -> bool {
  if (emit_tuples_.empty()) return false;
  *tuple = emit_tuples_.top();
  *rid = emit_tuples_.top().GetRid();
  emit_tuples_.pop();
  return true;
}

auto TopNExecutor::GetNumInHeap() -> size_t { return emit_tuples_.size(); };

}  // namespace bustub
