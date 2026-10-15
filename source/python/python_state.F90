! Native, typed Python configuration. No input deck or grid file is used.
module cfl3d_python_state
  implicit none
  integer, allocatable :: py_dims(:,:), py_counts(:,:), py_bc(:,:)
  integer, allocatable :: py_blocks(:,:), py_stages(:,:), py_connections(:,:,:)
  real(8), allocatable :: py_bcdata(:,:), py_xyz(:,:), py_q(:,:), py_history(:,:), py_surface(:,:)
  real(8), allocatable :: py_wall(:,:),py_wall_forces(:,:)
  integer, allocatable :: py_walloffset(:)
  integer, allocatable :: py_offsets(:), py_qoffsets(:)
  real(8) :: py_flow(7), py_reference(6), py_time(6), py_keywords(9)
  integer :: py_run(8), py_mg(5), py_scheme(6), py_flux(3), py_smoothing(5,5)
  real(8) :: py_rkap(3), py_final_alpha=0.0d0
  real(8) :: py_force_angles(2)=0.0d0
  integer :: py_force_angle_valid=0
  integer :: py_ngrid=0, py_nbc=0, py_nconn=0, py_nhistory=0
  integer :: py_status=0, py_used=0
  integer :: py_comm=-1, py_force_zone=0
  logical :: py_write_restart=.true.
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

  subroutine py_select_force_zone(block,zone_blocks)
    integer, intent(in) :: block,zone_blocks(*)
    integer :: g
    py_force_zone=0
    do g=1,py_ngrid
      if (block == zone_blocks(g)) py_force_zone=g
    end do
  end subroutine

  subroutine py_record_surface(face,segment,raw,i2d,ialph,alpha,beta)
    integer, intent(in) :: face,segment,i2d,ialph
    real(8), intent(in) :: raw(10)
    real(8), intent(in) :: alpha,beta
    real(8) :: values(10),temp,yref,zref
    integer :: idx
    if (py_force_zone == 0) return  ! Coarse-grid forces are not output surfaces.
    ! CL control updates alpha after force evaluation; retain the actual force axes.
    py_force_angles=(/alpha,beta/); py_force_angle_valid=1
    idx=py_bcindex(py_force_zone,face,segment)
    ! raw: CL, CD, CX, CY, CZ, Cmx, Cmy, Cmz, CDp, CDv before Sref/length scaling.
    values=raw
    ! Match force.F's 2D moment suppression and rotation to input coordinates.
    if (i2d == 1) then
      values(6)=0; values(8)=0
    end if
    yref=py_reference(2); zref=py_reference(3)
    if (ialph > 0) then
      temp=values(8); values(8)=-values(7); values(7)=temp
      temp=values(5); values(5)=-values(4); values(4)=temp
      yref=py_reference(3); zref=py_reference(2)
    end if
    values=values/py_reference(1)
    values(6)=values(6)/py_reference(3)
    values(7)=values(7)/yref
    values(8)=values(8)/zref
    ! Only the finest block's owner writes this row. Each evaluation replaces it.
    py_surface(:,idx)=values
  end subroutine

  subroutine py_record_wall(face,segment,a,b,pressure,density,cf,area,volume,cbar,ialph,force,viscous)
    integer, intent(in) :: face,segment,a,b,ialph
    real(8), intent(in) :: pressure,density,cf(3),area,volume,cbar
    real(8), intent(in) :: force(3),viscous(3)
    real(8) :: values(6),tratio,mu,temp,distance
    real(8) :: loads(6)
    integer :: idx,p,code
    if (py_force_zone == 0) return
    idx=py_bcindex(py_force_zone,face,segment)
    code=abs(py_bc(3,idx))
    if (.not.(code==1005.or.code==1006.or.code==2004.or.code==2014.or.&
              code==2024.or.code==2034.or.code==2016)) return
    p=py_walloffset(idx)+(a-py_bc(4,idx))+(py_bc(5,idx)-py_bc(4,idx))*(b-py_bc(6,idx))
    tratio=py_keywords(5)*pressure/density
    values(1)=(py_keywords(5)*pressure-1.d0)*2.d0/(py_keywords(5)*py_flow(1)**2)
    values(2)=tratio*py_flow(5)/1.8d0  ! Wall temperature in K, not adjacent-cell temperature.
    values(3:5)=cf
    ! Keep the native integrated cell loads: no reconstruction from Cp/normals is needed later.
    loads(1:3)=force-viscous; loads(4:6)=viscous
    if (ialph > 0) then
      temp=values(5); values(5)=-values(4); values(4)=temp
      temp=loads(3); loads(3)=-loads(2); loads(2)=temp
      temp=loads(6); loads(6)=-loads(5); loads(5)=temp
    end if
    values(6)=0
    if (area > 0.d0 .and. code >= 2000) then
      ! First-cell normal distance, consistent with the native wall traction V/(2*A).
      ! Unlike legacy nodal y+ statistics, this is a face-cell value without averaging.
      distance=0.5d0*volume/area
      mu=tratio**1.5d0*(1.d0+cbar/py_flow(5))/(tratio+cbar/py_flow(5))
      values(6)=distance*py_flow(4)*1.d6*sqrt(0.5d0*sqrt(sum(cf**2))*density)/mu
    end if
    py_wall(:,p)=values
    py_wall_forces(:,p)=loads/py_reference(1)
  end subroutine

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

subroutine py_api_version(version)
  implicit none
  integer, intent(out) :: version
  version=5
end subroutine

subroutine py_get_force_angles(angles)
  use cfl3d_python_state, only: py_force_angles
  implicit none
  real(8), intent(out) :: angles(2)
  angles=py_force_angles
end subroutine

subroutine py_set_output_flags(write_restart)
  use cfl3d_python_state, only: py_write_restart
  implicit none
  integer, intent(in) :: write_restart
  py_write_restart=write_restart /= 0
end subroutine

subroutine py_get_wall_data(values,ncells)
  use cfl3d_python_state, only: py_wall
  implicit none
  integer, intent(in) :: ncells
  real(8), intent(out) :: values(6,ncells)
  values=py_wall
end subroutine

subroutine py_get_wall_forces(values,ncells)
  use cfl3d_python_state, only: py_wall_forces
  implicit none
  integer, intent(in) :: ncells
  real(8), intent(out) :: values(6,ncells)
  values=py_wall_forces
end subroutine

subroutine py_get_surface_coefficients(values,nbc)
  use cfl3d_python_state, only: py_surface
  implicit none
  integer, intent(in) :: nbc
  real(8), intent(out) :: values(10,nbc)
  values=py_surface
end subroutine

subroutine py_get_final_alpha(alpha_degrees)
  use cfl3d_python_state, only: py_final_alpha
  implicit none
  real(8), intent(out) :: alpha_degrees
  alpha_degrees=py_final_alpha
end subroutine

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
  integer :: ierr,rank,owner,local_owner
  real(8), allocatable :: gathered(:,:)
  ! Each cell is filled only on its owning worker; host/other workers keep zero.
  ! SUM therefore assembles the solution without counting a cell twice.
  ! This returns full arrays on every rank, so Python storage is not distributed.
  allocate(gathered(size(py_q,1),size(py_q,2)))
  call MPI_Allreduce(py_q,gathered,size(py_q),MPI_DOUBLE_PRECISION,MPI_SUM,py_comm,ierr)
  py_q=gathered
  deallocate(gathered)
  allocate(gathered(size(py_surface,1),size(py_surface,2)))
  call MPI_Allreduce(py_surface,gathered,size(py_surface),MPI_DOUBLE_PRECISION,MPI_SUM,py_comm,ierr)
  py_surface=gathered
  deallocate(gathered)
  allocate(gathered(size(py_wall,1),size(py_wall,2)))
  call MPI_Allreduce(py_wall,gathered,size(py_wall),MPI_DOUBLE_PRECISION,MPI_SUM,py_comm,ierr)
  py_wall=gathered
  deallocate(gathered)
  allocate(gathered(size(py_wall_forces,1),size(py_wall_forces,2)))
  call MPI_Allreduce(py_wall_forces,gathered,size(py_wall_forces),MPI_DOUBLE_PRECISION,MPI_SUM,py_comm,ierr)
  py_wall_forces=gathered
  deallocate(gathered)
  call MPI_Comm_rank(py_comm,rank,ierr)
  local_owner=-1
  if (py_force_angle_valid /= 0) local_owner=rank
  call MPI_Allreduce(local_owner,owner,1,MPI_INTEGER,MPI_MAX,py_comm,ierr)
  if (owner >= 0) call MPI_Bcast(py_force_angles,2,MPI_DOUBLE_PRECISION,owner,py_comm,ierr)
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
                        time_control,run_control,mg,smoothing,stages,keywords,scheme,flux,rkap,&
                        ngrid,nbc,nconn,mseq,npoints,ncells,status)
  use cfl3d_python_state
  implicit none
  integer, intent(in) :: ngrid,nbc,nconn,mseq,npoints,ncells
  integer, intent(in) :: dims(3,ngrid),counts(6,ngrid),bcs(8,nbc),blocks(7,ngrid)
  integer, intent(in) :: connections(10,2,nconn),run_control(8),mg(5),stages(4,mseq)
  integer, intent(in) :: scheme(6),flux(3),smoothing(5,5)
  real(8), intent(in) :: bcdata(12,nbc),flow(7),reference(6),time_control(6),keywords(9),rkap(3)
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
  if (allocated(py_surface)) deallocate(py_surface)
  if (allocated(py_wall)) deallocate(py_wall,py_walloffset,py_wall_forces)
  allocate(py_walloffset(nbc))
  p=1
  do g=1,nbc
    py_walloffset(g)=p
    p=p+(bcs(5,g)-bcs(4,g))*(bcs(7,g)-bcs(6,g))
  end do
  allocate(py_wall(6,p-1))
  allocate(py_wall_forces(6,p-1))
  py_wall_forces=0
  py_wall=0
  allocate(py_surface(10,nbc))
  py_surface=0; py_force_zone=0
  py_ngrid=ngrid; py_nbc=nbc; py_nconn=nconn
  py_dims=dims; py_counts=counts; py_bc=bcs; py_bcdata=bcdata
  py_blocks=blocks; py_connections=connections; py_stages=stages
  py_flow=flow; py_reference=reference; py_time=time_control
  py_force_angles=flow(2:3)*(acos(-1.d0)/180.d0); py_force_angle_valid=0
  py_run=run_control; py_mg=mg; py_keywords=keywords
  py_smoothing=smoothing; py_final_alpha=flow(2)
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
