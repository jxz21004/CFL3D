/* Compile against the selected CGNS headers to expose its actual ABI. */
#include <string.h>
#include <cgnslib.h>

_Static_assert(sizeof(CGNS_ENUMT(BCType_t)) == sizeof(int), "CGNS enum ABI");

int cfl3d_cgns_index_bytes(void) { return sizeof(cgsize_t); }
int cfl3d_cgns_has_hdf5(void) { return CG_BUILD_HDF5; }
/* This reference also keeps libcgns in the bridge's link dependencies. */
const char *cfl3d_cgns_error(void) { return cg_get_error(); }

int cfl3d_cgns_constant(const char *name) {
#define VALUE(x) if (!strcmp(name, #x)) return x
#define ENUM(x) if (!strcmp(name, #x)) return CGNS_ENUMV(x)
    VALUE(CG_MODE_READ);
    VALUE(CG_MODE_WRITE);
    VALUE(CG_MODE_MODIFY);
    VALUE(CG_FILE_ADF);
    VALUE(CG_FILE_HDF5);
    VALUE(CG_NODE_NOT_FOUND);
    VALUE(NofValidBCTypes);
    ENUM(Structured);
    ENUM(RealDouble);
    ENUM(RealSingle);
    ENUM(Integer);
    ENUM(LongInteger);
    ENUM(PointRange);
    ENUM(PointList);
    ENUM(Vertex);
    ENUM(CellCenter);
    ENUM(Dirichlet);
    ENUM(Neumann);
    return -1;
}
