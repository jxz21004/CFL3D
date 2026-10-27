! Native, typed Python configuration. No input deck or grid file is used.
module cfl3d_python_state
  implicit none
  integer, allocatable :: py_dims(:,:), py_counts(:,:), py_bc(:,:)
  integer, allocatable :: py_blocks(:,:), py_stages(:,:), py_connections(:,:,:)
  real(8), allocatable :: py_bcdata(:,:), py_xyz(:,:), py_q(:,:), py_history(:,:), py_surface(:,:)
  real(8), allocatable :: py_wall(:,:),py_wall_forces(:,:)
  integer, allocatable :: py_walloffset(:)
  integer, allocatable :: py_offsets(:), py_qoffsets(:), py_zone_owners(:)
  real(8) :: py_flow(7), py_reference(6), py_time(6), py_keywords(15)
  integer :: py_run(8), py_mg(5), py_scheme(6), py_flux(3), py_smoothing(5,5)
  real(8) :: py_rkap(3), py_final_alpha=0.0d0
  real(8) :: py_force_angles(2)=0.0d0
  integer :: py_force_angle_valid=0
  integer :: py_ngrid=0, py_nbc=0, py_nconn=0, py_nhistory=0
  integer :: py_status=0, py_used=0
  integer :: py_comm=-1, py_force_zone=0
  ! Native force integration runs every cycle; detailed output is captured only
  ! at a Python solve boundary while the finest-grid wall state is still live.
  logical :: py_capture_surface=.false.
  logical :: py_write_restart=.true.
  character(len=60) :: py_output_stem=''
  logical :: py_append_output=.false.
  logical :: py_override_restart_alpha=.false.
  integer :: py_iteration_offset=0
  ! Resident driver, enabled by options['resident'] or the explicit native API.
  ! Default public solves retain the established restart-based path.
  logical :: py_resident_enabled=.false.,py_resident_paused=.false.
  logical :: py_resuming=.false.,py_resident_closing=.false.
  logical :: py_resident_open_ended=.false.
  logical :: py_resident_output_changed=.false.
  ! Staged owned coordinates exist only between an explicit update and resume.
  logical :: py_geometry_pending=.false.
  integer :: py_geometry_updates=0
  integer, allocatable :: py_geometry_offsets(:)
  real(8), allocatable :: py_geometry_xyz(:,:)
  integer :: py_resident_stop=0,py_resident_done=0
  integer :: py_resident_written=0
  integer :: py_resident_initializations=0,py_resident_capacity=0
  integer(kind=8) :: py_resident_work_bytes=0
  real(8) :: py_resident_cfl=0.d0
  interface py_grow_history
    module procedure py_grow_history_1,py_grow_history_2,py_grow_history_3,py_grow_history_4,py_grow_history_int
  end interface
contains
  subroutine py_write_resident_history(capacity,nummem,rms,cl,cd,cx,cy,cz,cmx,cmy,cmz,turres,nneg)
    integer, intent(in) :: capacity,nummem,nneg(capacity,nummem)
    real(8), intent(in) :: rms(capacity),cl(capacity),cd(capacity),cx(capacity),cy(capacity),cz(capacity)
    real(8), intent(in) :: cmx(capacity),cmy(capacity),cmz(capacity),turres(capacity,nummem)
    integer :: n,l
    ! Standalone histout runs only at final cleanup and can modify zero RMS
    ! entries. Emit completed rows at each resident boundary without changing
    ! authoritative histories or writing them a second time at close.
    write(12,'(a)') '# it log(res) CL CD CX CY CZ Cmx Cmy Cmz'
    write(13,'(a)') '# it log(turbulence residuals...) negative-counts...'
    do n=py_resident_written+1,py_resident_done
      write(12,'(i12,9es16.8)') n+py_iteration_offset,log10(merge(rms(n),1.d0,rms(n)>0.d0)), &
                               cl(n),cd(n),cx(n),cy(n),cz(n),cmx(n),cmy(n),cmz(n)
      if (maxval(py_blocks(5:7,:)) > 2) then
        write(13,'(i12,*(es16.8,1x))',advance='no') n+py_iteration_offset, &
                         (log10(merge(turres(n,l),1.d0,turres(n,l)>0.d0)),l=1,nummem)
        write(13,'(1x,*(i12,1x))') (nneg(n,l),l=1,nummem)
      end if
    end do
    py_resident_written=py_resident_done
  end subroutine

  ! Only iteration histories grow. Geometry, native work arrays and the MG
  ! hierarchy keep their addresses and contents across an extension. Move one
  ! history at a time to bound temporary memory to the largest history array.
  subroutine py_grow_history_1(values,capacity)
    real(8), allocatable, intent(inout) :: values(:)
    integer, intent(in) :: capacity
    real(8), allocatable :: grown(:)
    if (size(values) >= capacity) return
    allocate(grown(capacity)); grown=0.d0
    grown(:size(values))=values
    call move_alloc(grown,values)
  end subroutine

  subroutine py_grow_history_2(values,capacity)
    real(8), allocatable, intent(inout) :: values(:,:)
    integer, intent(in) :: capacity
    real(8), allocatable :: grown(:,:)
    if (size(values,1) >= capacity) return
    allocate(grown(capacity,size(values,2))); grown=0.d0
    grown(:size(values,1),:)=values
    call move_alloc(grown,values)
  end subroutine

  subroutine py_grow_history_int(values,capacity)
    integer, allocatable, intent(inout) :: values(:,:)
    integer, intent(in) :: capacity
    integer, allocatable :: grown(:,:)
    if (size(values,1) >= capacity) return
    allocate(grown(capacity,size(values,2))); grown=0
    grown(:size(values,1),:)=values
    call move_alloc(grown,values)
  end subroutine

  subroutine py_grow_history_3(values,capacity)
    real(8), allocatable, intent(inout) :: values(:,:,:)
    integer, intent(in) :: capacity
    real(8), allocatable :: grown(:,:,:)
    if (size(values,3) >= capacity) return
    allocate(grown(size(values,1),size(values,2),capacity)); grown=0.d0
    grown(:,:,:size(values,3))=values
    call move_alloc(grown,values)
  end subroutine

  subroutine py_grow_history_4(values,capacity)
    real(8), allocatable, intent(inout) :: values(:,:,:,:)
    integer, intent(in) :: capacity
    real(8), allocatable :: grown(:,:,:,:)
    if (size(values,1) >= capacity) return
    allocate(grown(capacity,size(values,2),size(values,3),size(values,4))); grown=0.d0
    grown(:size(values,1),:,:,:)=values
    call move_alloc(grown,values)
  end subroutine

  function py_output_name(legacy) result(name)
    character(len=*), intent(in) :: legacy
    character(len=80) :: name
    name=legacy
    if (len_trim(py_output_stem)==0) return
    if (index(legacy,'cfl3d.')==1) then
      name=trim(py_output_stem)//legacy(6:)
    else
      name=trim(py_output_stem)//'_'//trim(legacy)
    end if
  end function

  subroutine py_open_text(unit,name)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: name
    logical :: connected
    ! A few legacy diagnostic units survive a solve. Close them explicitly before
    ! reopening the same filename with a different STATUS/POSITION on continuation.
    inquire(unit=unit,opened=connected)
    if (connected) close(unit)
    ! Open the final name before computing, so live logs need no later copy.
    if (py_append_output) then
      open(unit=unit,file=name,form='formatted',status='unknown',position='append')
    else
      open(unit=unit,file=name,form='formatted',status='replace')
    end if
  end subroutine

  subroutine py_flush_resident_output()
    integer :: unit
    logical :: connected
    do unit=11,99
      ! Flush connected native diagnostics at a Python boundary. Do not create
      ! a fort.N file by flushing an unconnected worker unit.
      inquire(unit=unit,opened=connected)
      if (connected) flush(unit)
    end do
  end subroutine

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
    if (.not.py_capture_surface .or. py_force_zone == 0) return
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
    if (.not.py_capture_surface .or. py_force_zone == 0) return
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

  subroutine py_copy_updated_grid(x,y,z,jd,kd,id,grid,ialph)
    integer, intent(in) :: jd,kd,id,grid,ialph
    real(8), intent(out) :: x(jd,kd,id),y(jd,kd,id),z(jd,kd,id)
    integer :: i,j,k,p
    do k=1,kd
      do j=1,jd
        do i=1,id
          p=py_geometry_offsets(grid)+(i-1)+id*((j-1)+jd*(k-1))
          x(j,k,i)=py_geometry_xyz(p,1)
          if (ialph == 0) then
            y(j,k,i)=py_geometry_xyz(p,2); z(j,k,i)=py_geometry_xyz(p,3)
          else
            y(j,k,i)=-py_geometry_xyz(p,3); z(j,k,i)=py_geometry_xyz(p,2)
          end if
        end do
      end do
    end do
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

  subroutine py_partition_results(zone_blocks,block_owners,rank)
    integer, intent(in) :: zone_blocks(*),block_owners(*),rank
    integer :: g,c,p,code
    ! Use the actual native load balancer, including its finest/coarse mapping.
    ! Do not allocate a global cell array and slice it after the solve.
    if (allocated(py_q)) deallocate(py_q)
    c=1
    do g=1,py_ngrid
      py_zone_owners(g)=block_owners(zone_blocks(g))
      py_qoffsets(g)=0
      if (py_zone_owners(g) /= rank) cycle
      py_qoffsets(g)=c
      c=c+product(py_dims(:,g)-1)
    end do
    allocate(py_q(c-1,5))
    py_q=0
    ! Wall-cell diagnostics follow the same zone owner. Interface/farfield
    ! segments have no detailed wall buffer, even in serial builds.
    deallocate(py_wall,py_wall_forces)
    p=1
    do g=1,py_nbc
      py_walloffset(g)=0
      code=abs(py_bc(3,g))
      if (py_zone_owners(py_bc(1,g)) /= rank) cycle
      if (.not.(code==1005.or.code==1006.or.code==2004.or.code==2014.or.&
                code==2024.or.code==2034.or.code==2016)) cycle
      py_walloffset(g)=p
      p=p+(py_bc(5,g)-py_bc(4,g))*(py_bc(7,g)-py_bc(6,g))
    end do
    allocate(py_wall(6,p-1),py_wall_forces(6,p-1))
    py_wall=0; py_wall_forces=0
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
  version=17
end subroutine

subroutine py_set_restart_controls(override_alpha)
  use cfl3d_python_state, only: py_override_restart_alpha
  implicit none
  integer, intent(in) :: override_alpha
  py_override_restart_alpha=override_alpha/=0
end subroutine

subroutine py_set_output_names(stem,append_output,iteration_offset)
  use cfl3d_python_state
  implicit none
  character(len=*), intent(in) :: stem
  integer, intent(in) :: append_output,iteration_offset
  py_output_stem=stem
  py_append_output=append_output/=0
  py_iteration_offset=iteration_offset
  if (py_resident_enabled .and. py_resident_paused) py_resident_output_changed=.true.
end subroutine

subroutine py_rotate_resident_output()
  use cfl3d_python_state
  implicit none
  integer :: nnodes,myhost,myid,mycomm
  common /mydist2/ nnodes,myhost,myid,mycomm
  character(80) :: grid,plt3dg,plt3dq,output,residual,turbres,blomx,output2,printout,pplunge, &
                  ovrlap,patch,restrt,subres,subtur,grdmov,alphahist,errfile,preout,aeinp,aeout,sdhist,avgg,avgq
  common /filenam/ grid,plt3dg,plt3dq,output,residual,turbres,blomx,output2,printout,pplunge, &
                  ovrlap,patch,restrt,subres,subtur,grdmov,alphahist,errfile,preout,aeinp,aeout,sdhist,avgg,avgq
  if (.not.py_resident_output_changed) return
  py_resident_output_changed=.false.
  call py_flush_resident_output()
  ! Rotate diagnostics only. Setup's patch/overlap files remain attached to
  ! their original units; reopening those would disturb the retained context.
  errfile=py_output_name('cfl3d.error')
  call py_open_text(99,errfile)
  if (myid /= myhost) return
  output=py_output_name('cfl3d.out'); call py_open_text(11,output)
  residual=py_output_name('cfl3d.res'); call py_open_text(12,residual)
  turbres=py_output_name('cfl3d.turres'); call py_open_text(13,turbres)
  blomx=py_output_name('cfl3d.blomax'); call py_open_text(14,blomx)
  output2=py_output_name('cfl3d.out15'); call py_open_text(15,output2)
  printout=py_output_name('cfl3d.prout'); call py_open_text(17,printout)
  pplunge=py_output_name('cfl3d.out20'); call py_open_text(20,pplunge)
  subres=py_output_name('cfl3d.subit_res'); call py_open_text(23,subres)
  subtur=py_output_name('cfl3d.subit_turres'); call py_open_text(24,subtur)
  grdmov=py_output_name('cfl3d.dynamic_patch'); call py_open_text(25,grdmov)
  alphahist=py_output_name('cfl3d.alpha'); call py_open_text(27,alphahist)
  write(11,'(/,a,i12)') 'Python resident continuation; completed native cycles: ',py_resident_done
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
  if (.not.initialized .or. finalized .or. py_used /= 0 .or. py_resident_enabled) return
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
  ! Detailed fields remain on their owner. Reduce only one integral row per
  ! boundary; these small diagnostics support noncollective load queries.
  allocate(gathered(size(py_surface,1),size(py_surface,2)))
  call MPI_Allreduce(py_surface,gathered,size(py_surface),MPI_DOUBLE_PRECISION,MPI_SUM,py_comm,ierr)
  py_surface=gathered
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
#ifdef DIST_MPI
  include 'mpif.h'
  integer :: rank,ierr
#endif
  integer, intent(in) :: ngrid,nbc,nconn,mseq,npoints,ncells
  integer, intent(in) :: dims(3,ngrid),counts(6,ngrid),bcs(8,nbc),blocks(7,ngrid)
  integer, intent(in) :: connections(10,2,nconn),run_control(8),mg(5),stages(4,mseq)
  integer, intent(in) :: scheme(6),flux(3),smoothing(5,5)
  real(8), intent(in) :: bcdata(12,nbc),flow(7),reference(6),time_control(6),keywords(15),rkap(3)
  integer, intent(out) :: status
  integer :: g,p,c
  status=0
  if (py_used /= 0) then
    status=1
    return
  end if
  if (py_resident_enabled) then
    status=3
    return
  end if
  if (allocated(py_dims)) then
    deallocate(py_dims,py_counts,py_bc,py_blocks,py_stages,py_connections)
    deallocate(py_bcdata,py_offsets,py_qoffsets,py_zone_owners)
    if (allocated(py_xyz)) deallocate(py_xyz)
    if (allocated(py_q)) deallocate(py_q)
  end if
  if (allocated(py_history)) deallocate(py_history)
  if (allocated(py_surface)) deallocate(py_surface)
  if (allocated(py_wall)) deallocate(py_wall,py_wall_forces)
  if (allocated(py_walloffset)) deallocate(py_walloffset)
  allocate(py_walloffset(nbc))
  py_walloffset=0
  allocate(py_wall(6,0),py_wall_forces(6,0))
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
  ! Only the native host reads coordinates through rp3d. Workers receive
  ! their grid via the existing native MPI setup, not a replicated f2py copy.
#ifdef DIST_MPI
  call MPI_Comm_rank(py_comm,rank,ierr)
  if (rank == 0) then
    allocate(py_xyz(npoints,3))
  else
    allocate(py_xyz(0,3))
  end if
#else
  allocate(py_xyz(npoints,3))
#endif
  allocate(py_q(0,5),py_offsets(ngrid),py_qoffsets(ngrid),py_zone_owners(ngrid))
  py_zone_owners=0
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
  ! A worker deliberately has a zero-sized coordinate input buffer.
  if (size(py_xyz,1) == 0) return
  py_xyz=xyz
end subroutine

subroutine py_get_partition(owners,ncells,nwallcells,ngrid)
  use cfl3d_python_state
  implicit none
  integer, intent(in) :: ngrid
  integer, intent(out) :: owners(ngrid),ncells,nwallcells
  owners=py_zone_owners
  ncells=size(py_q,1)
  nwallcells=size(py_wall,2)
end subroutine

subroutine py_release_buffers()
  use cfl3d_python_state
  implicit none
  if (py_resident_enabled) return
  ! Python has taken ownership of the returned state. Keep no second native
  ! copy between calls; configuration and callbacks rebuild these buffers.
  if (allocated(py_xyz)) deallocate(py_xyz)
  if (allocated(py_q)) deallocate(py_q)
  if (allocated(py_wall)) deallocate(py_wall,py_wall_forces)
  if (allocated(py_surface)) deallocate(py_surface)
  if (allocated(py_history)) deallocate(py_history)
end subroutine

subroutine py_start_resident(cycles,open_ended,status)
  use cfl3d_python_state
  implicit none
  integer, intent(in) :: cycles,open_ended
  integer, intent(out) :: status
  integer :: last,startup
  integer(kind=8) :: total
  status=1
  if (py_used /= 0 .or. py_resident_enabled .or. cycles < 1) return
  if (.not.allocated(py_xyz) .or. .not.allocated(py_stages)) return
  if (py_time(1) >= 0.d0 .or. py_time(2) /= 0.d0) return
  ! Finish native coarse-to-fine startup before returning any finest-grid
  ! readout. cycles counts all startup cycles plus the requested fine cycles.
  last=size(py_stages,2)
  if (last < 1 .or. last > 5) return
  total=sum(int(py_stages(1,:),kind=8))
  if (total > int(huge(cycles),kind=8)) return
  startup=int(total)-py_stages(1,last)
  if (cycles <= startup) return
  if (open_ended /= 0 .and. open_ended /= 1) return
  if (open_ended == 0 .and. cycles > int(total)) return
  py_resident_open_ended=(open_ended == 1)
  if (py_resident_open_ended) py_stages(1,last)=max(py_stages(1,last),cycles-startup)
  py_resident_enabled=.true.; py_resident_paused=.false.
  py_resident_output_changed=.false.
  py_resuming=.false.; py_resident_closing=.false.
  py_resident_stop=cycles; py_resident_done=0
  py_resident_written=0
  py_geometry_pending=.false.; py_geometry_updates=0
  if (allocated(py_geometry_xyz)) deallocate(py_geometry_xyz,py_geometry_offsets)
  py_resident_initializations=0; py_resident_capacity=0
  py_resident_work_bytes=0; py_resident_cfl=0.d0
  py_write_restart=.false.
  status=0
end subroutine

subroutine py_advance_resident(cycles,cfl,status,nrows)
  use cfl3d_python_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  integer, intent(in) :: cycles
  real(8), intent(in) :: cfl
  integer, intent(out) :: status,nrows
  integer :: last,startup
  status=1; nrows=py_nhistory
  if (.not.py_resident_enabled .or. .not.py_resident_paused .or. py_used /= 0) return
  if (cycles < 1 .or. .not.ieee_is_finite(cfl) .or. .not.(cfl >= 0.d0)) return
  ! Check before adding: the original driver uses default integer counters.
  if (cycles > huge(cycles)-py_resident_done) return
  if (py_resident_open_ended) then
    last=size(py_stages,2)
    startup=sum(py_stages(1,:last-1))
    py_stages(1,last)=max(py_stages(1,last),py_resident_done+cycles-startup)
  else
    if (cycles > sum(py_stages(1,:))-py_resident_done) return
  end if
  py_resident_stop=py_resident_done+cycles
  py_resident_cfl=cfl
  py_resuming=.true.; py_resident_paused=.false.
  py_surface=0; py_force_angle_valid=0
  call py_solve(status,nrows)
end subroutine

subroutine py_close_resident(status)
  use cfl3d_python_state
  implicit none
  integer, intent(out) :: status
  integer :: nrows
  status=0
  if (.not.py_resident_enabled) return
  if (py_used /= 0) then
    status=1
    return
  end if
  if (py_resident_initializations == 0) then
    py_resident_enabled=.false.
    return
  end if
  if (.not.py_resident_paused) then
    status=1
    return
  end if
  py_resident_closing=.true.; py_resuming=.true.
  py_resident_paused=.false.
  call py_solve(status,nrows)
  py_resident_closing=.false.
  py_geometry_pending=.false.
  if (allocated(py_geometry_xyz)) deallocate(py_geometry_xyz,py_geometry_offsets)
end subroutine

subroutine py_resident_info(enabled,paused,cycles,initializations,capacity,work_bytes,coordinate_values)
  use cfl3d_python_state
  implicit none
  integer, intent(out) :: enabled,paused,cycles,initializations,capacity
  integer(kind=8), intent(out) :: work_bytes,coordinate_values
  enabled=0; paused=0
  if (py_resident_enabled) enabled=1
  if (py_resident_paused) paused=1
  cycles=py_resident_done; initializations=py_resident_initializations
  capacity=py_resident_capacity; work_bytes=py_resident_work_bytes
  coordinate_values=0
  if (allocated(py_xyz)) coordinate_values=size(py_xyz,kind=8)
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
  if (py_resident_paused) then
    status=2; nrows=py_nhistory
    return
  end if
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

subroutine py_stage_geometry(xyz,rank,npoints,status)
  use cfl3d_python_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  integer, intent(in) :: rank,npoints
  real(8), intent(in) :: xyz(npoints,3)
  integer, intent(out) :: status
  integer :: g,p
  status=1
  if (.not.py_resident_paused .or. py_used /= 0) return
  p=0
  do g=1,py_ngrid
    if (py_zone_owners(g) == rank) p=p+product(py_dims(:,g))
  end do
  if (p /= npoints .or. .not.all(ieee_is_finite(xyz))) return
  if (allocated(py_geometry_xyz)) deallocate(py_geometry_xyz,py_geometry_offsets)
  allocate(py_geometry_xyz(npoints,3),py_geometry_offsets(py_ngrid))
  py_geometry_xyz=xyz; py_geometry_offsets=0
  p=1
  do g=1,py_ngrid
    if (py_zone_owners(g) /= rank) cycle
    py_geometry_offsets(g)=p
    p=p+product(py_dims(:,g))
  end do
  py_geometry_pending=.true.
  status=0
end subroutine

subroutine py_geometry_info(updates)
  use cfl3d_python_state, only: py_geometry_updates
  implicit none
  integer, intent(out) :: updates
  updates=py_geometry_updates
end subroutine
