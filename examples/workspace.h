//
//  workspace.h - the ulm_lam workspace convention, for the C++ drivers.
//
//  A generated function that needs temporaries takes two trailing
//  parameters (int8_t *work, int64_t *workspace_info):
//
//    *workspace_info == -1  query: the function sets the required size
//                           (bytes) and returns
//    *workspace_info ==  0  the function allocates one block itself
//    *workspace_info  >  0  the size of work; too small: the function returns
//                           info = -i (i: the position of workspace_info)
//
//  Workspace ws([](int8_t *w, int64_t *i) { f(..., w, i); });  queries and
//  allocates; then call f(..., ws.work, ws.info()).
//
#ifndef ULM_LAM_EXAMPLES_WORKSPACE_H
#define ULM_LAM_EXAMPLES_WORKSPACE_H

#include <cstdint>
#include <cstdlib>

struct Workspace
{
    int8_t *work = nullptr;
    int64_t size = 0;
    int64_t current = 0;

    template <typename Call> explicit Workspace(Call call)
    {
	int64_t query = -1;
	call(nullptr, &query);
	size = query;
	work = static_cast<int8_t *>(
	    std::aligned_alloc(64, size > 0 ? (size + 63) / 64 * 64 : 64));
    }
    ~Workspace()
    {
	std::free(work);
    }
    Workspace(const Workspace &) = delete;
    Workspace &
    operator=(const Workspace &) = delete;

    /// *workspace_info for a call: the size of the workspace
    int64_t *
    info()
    {
	current = size;
	return &current;
    }
};

#endif
