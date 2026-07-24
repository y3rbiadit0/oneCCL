Group calls 
***********

.. doxygengroup:: group_calls
   :project: oneccl
   :content-only:
   :no-link:

NCCL backend behavior
=====================

With the NCCL backend, the outermost group starts and ends an NCCL group. Nested group calls are
allowed, but inner ``group_end`` calls do not submit the operations. Submission occurs when the
outermost ``group_end`` returns.

NCCL may defer placing grouped operations on their CUDA streams until the group ends. Events
returned by grouped operations therefore remain incomplete until the outermost ``group_end``.
Calling ``wait`` or accessing the native handle of one of these events before that point is not
supported. After submission, events track a SYCL barrier placed after the NCCL work on each stream.

The NCCL backend requires in-order SYCL queues. The NCCL barrier implementation is not supported
inside a group because it uses temporary device storage that must remain valid until submission.
