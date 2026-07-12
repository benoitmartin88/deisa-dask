# Plan: Execute tasks on bridge when possible (Issue #95)

## IMPLEMENTED ✓

### 1. HandshakeActor task hints (handshake.py) ✓
- Added `task_hints: Dict` attribute to HandshakeActor.__init__
- Added `set_task_hints(array_name, hints)` method
- Added `get_task_hints(array_name)` method  
- Added `get_task_hints_dict()` method
- Added wrapper methods to Handshake class

### 2. Callback AST analysis (deisa.py) ✓
- Added `inspect` and `ast` imports
- Added `_analyze_callback_for_operations()` method
- Added `_extract_operation_from_call()` helper
- Added `_is_reducible_operation()` helper
- Modified `_register_callback_impl()` to call analysis and store hints

### 3. Bridge operation executor (bridge.py) ✓
- Added `task_hints` broadcast in `__init__` (only rank 0 fetches, all get via bcast)
- Added `_execute_operations_on_chunk()` method
- Added `_combine_reduction_partials()` method
- Modified `send()` to:
  - Call `_execute_operations_on_chunk()` before scatter
  - Include `partials` in `to_send`
  - Call `_combine_reduction_partials()` after gather
  - Include `precomputed` in final payload
- Modified `_direct_send()` to accept and include partials

### 4. Topic handler integration (deisa.py) ✓
- Modified `_make_topic_handler()` to attach `precomputed` to dask array

---

## Supported Reducible Operations
- `sum()` → partial: chunk.sum(), combine: sum(all)
- `max()` → partial: chunk.max(), combine: max(all)
- `min()` → partial: chunk.min(), combine: min(all)
- `mean()` → partial: sum, count, combine: sum_sum / sum_count