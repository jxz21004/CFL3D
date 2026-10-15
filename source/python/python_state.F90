! Native, typed Python configuration. No input deck or grid file is used.
module cfl3d_python_state
  implicit none
  integer, allocatable :: py_dims(:,:), py_counts(:,:), py_bc(:,:)
  integer, allocatable :: py_blocks(:,:), py_stages(:,:), py_connections(:,:,:)
  real(8), allocatable :: py_bcdata(:,:), py_xyz(:,:), py_q(:,:), py_history(:,:)
  integer, allocatable :: py_offsets(:), py_qoffsets(:)
  real(8) :: py_flow(7), py_reference(6), py_time(6), py_keywords(5)
  integer :: py_run(8), py_mg(5), py_scheme(6), py_flux(3)
  real(8) :: py_rkap(3)
  integer :: py_ngrid=0, py_nbc=0, py_nconn=0, py_nhistory=0
  integer :: py_status=0, py_used=0
  integer :: py_comm=-1
contains
  integer function py_bcindex(grid,face,seg) result(idx)
    integer, intent(in) :: grid,face,seg
    integer :: f,g
    idx=seg
    do f=1,face-1
      idx=idx+sum(py_counts(f,:))
    end do
    do g=1,grid-1
      idx=idx+py_counts(face,g)
    end do
  end function

  subroutine py_copy_grid(x,y,z,jd,kd,id,grid,ialph)
    integer, intent(in) :: jd,kd,id,grid,ialph
    real(8), intent(out) :: x(jd,kd,id),y(jd,kd,id),z(jd,kd,id)
    integer :: i,j,k,p
    do k=1,kd
      do j=1,jd
        do i=1,id
          p=py_offsets(grid)+(i-1)+id*((j-1)+jd*(k-1))
          x(j,k,i)=py_xyz(p,1)
          if (ialph == 0) then
            y(j,k,i)=py_xyz(p,2)
            z(j,k,i)=py_xyz(p,3)
          else
            ! Match rp3d's y-up coordinate convention.
            y(j,k,i)=-py_xyz(p,3)
            z(j,k,i)=py_xyz(p,2)
          end if
        end do
      end do
    end do
  end subroutine

  subroutine py_copy_solution(q,jd,kd,id,grid)
    integer, intent(in) :: jd,kd,id,grid
    real(8), intent(in) :: q(jd,kd,id,5)
    integer :: i,j,k,p,v
    ! Cell-centered physical indices are 1..dimension-1 in all directions.
    do k=1,kd-1
      do j=1,jd-1
        do i=1,id-1
          p=py_qoffsets(grid)+(i-1)+(id-1)*((j-1)+(jd-1)*(k-1))
          do v=1,5
            py_q(p,v)=q(j,k,i,v)
          end do
        end do
      end do
    end do
  end subroutine
end module

subroutine py_build_info(mpi_enabled)
  implicit none
  integer, intent(out) :: mpi_enabled
#ifdef DIST_MPI
  mpi_enabled=1
#else
  mpi_enabled=0
#endif
end subroutine

subroutine py_set_communicator(comm,status)
  use cfl3d_python_state
  implicit none
#ifdef DIST_MPI
  include 'mpif.h'
  logical :: initialized,finalized
  integer :: ierr,nprocs
#endif
  integer, intent(in) :: comm
  integer, intent(out) :: status
  status=1
#ifdef DIST_MPI
  call MPI_Initialized(initialized,ierr)
  call MPI_Finalized(finalized,ierr)
  if (.not.initialized .or. finalized .or. py_used /= 0) return
  if (comm == MPI_COMM_NULL) return
  call MPI_Comm_size(comm,nprocs,ierr)
  if (ierr /= MPI_SUCCESS .or. nprocs < 2) return
  py_comm=comm
  status=0
#endif
end subroutine

subroutine py_collect_results()
  use cfl3d_python_state
  implicit none
#ifdef DIST_MPI
  include 'mpif.h'
  integer :: ierr
  real(8), allocatable :: gathered(:,:)
  ! Each cell is filled only on its owning worker; host/other workers keep zero.
  ! SUM therefore assembles the solution without counting a cell twice.
  ! This returns full arrays on every rank, so Python storage is not distributed.
  allocate(gathered(size(py_q,1),size(py_q,2)))
  call MPI_Allreduce(py_q,gathered,size(py_q),MPI_DOUBLE_PRECISION,MPI_SUM,py_comm,ierr)
  py_q=gathered
  deallocate(gathered)
  call MPI_Bcast(py_history,size(py_history),MPI_DOUBLE_PRECISION,0,py_comm,ierr)
#endif
end subroutine

subroutine py_abort(status)
  use cfl3d_python_state, only: py_comm
  implicit none
#ifdef DIST_MPI
  include 'mpif.h'
  integer :: ierr
#endif
  integer, intent(in) :: status
#ifdef DIST_MPI
  ! A local longjmp would strand peers inside legacy blocking MPI calls.
  call MPI_Abort(py_comm,max(1,abs(status)),ierr)
#endif
  call py_abort_jump(status)
end subroutine

subroutine py_configure(dims,counts,bcs,bcdata,blocks,connections,flow,reference,&
                        time_control,run_control,mg,stages,keywords,scheme,flux,rkap,&
                        ngrid,nbc,nconn,mseq,npoints,ncells,status)
  use cfl3d_python_state
  implicit none
  integer, intent(in) :: ngrid,nbc,nconn,mseq,npoints,ncells
  integer, intent(in) :: dims(3,ngrid),counts(6,ngrid),bcs(8,nbc),blocks(7,ngrid)
  integer, intent(in) :: connections(10,2,nconn),run_control(8),mg(5),stages(4,mseq)
  integer, intent(in) :: scheme(6),flux(3)
  real(8), intent(in) :: bcdata(12,nbc),flow(7),reference(6),time_control(6),keywords(5),rkap(3)
  integer, intent(out) :: status
  integer :: g,p,c
  status=0
  if (py_used /= 0) then
    status=1
    return
  end if
  if (allocated(py_dims)) then
    deallocate(py_dims,py_counts,py_bc,py_blocks,py_stages,py_connections)
    deallocate(py_bcdata,py_xyz,py_q,py_offsets,py_qoffsets)
  end if
  if (allocated(py_history)) deallocate(py_history)
  py_ngrid=ngrid; py_nbc=nbc; py_nconn=nconn
  py_dims=dims; py_counts=counts; py_bc=bcs; py_bcdata=bcdata
  py_blocks=blocks; py_connections=connections; py_stages=stages
  py_flow=flow; py_reference=reference; py_time=time_control
  py_run=run_control; py_mg=mg; py_keywords=keywords
  py_scheme=scheme; py_flux=flux; py_rkap=rkap
  allocate(py_xyz(npoints,3),py_q(ncells,5),py_offsets(ngrid),py_qoffsets(ngrid))
  py_xyz=0; py_q=0
  p=1; c=1
  do g=1,ngrid
    py_offsets(g)=p; py_qoffsets(g)=c
    p=p+product(dims(:,g)); c=c+product(dims(:,g)-1)
  end do
  if (p-1 /= npoints .or. c-1 /= ncells) status=2
  py_status=0; py_nhistory=0
end subroutine

subroutine py_set_coordinates(xyz,npoints)
  use cfl3d_python_state
  implicit none
  integer, intent(in) :: npoints
  real(8), intent(in) :: xyz(npoints,3)
  py_xyz=xyz
end subroutine

subroutine py_get_solution(q,ncells)
  use cfl3d_python_state
  implicit none
  integer, intent(in) :: ncells
  real(8), intent(out) :: q(ncells,5)
  q=py_q
end subroutine

subroutine py_get_history(history,nrows)
  use cfl3d_python_state
  implicit none
  integer, intent(in) :: nrows
  real(8), intent(out) :: history(nrows,8)
  history=py_history(1:nrows,:)
end subroutine

subroutine py_solve(status,nrows)
  use cfl3d_python_state
  implicit none
  integer, intent(out) :: status,nrows
  if (py_used /= 0 .or. .not.allocated(py_xyz)) then
    status=1; nrows=0
    return
  end if
  py_used=1
  call py_guarded_solve(status)
  nrows=py_nhistory
  ! An error may leave legacy allocations alive; only a successful run is reusable.
  if (status == 0) py_used=0
end subroutine
