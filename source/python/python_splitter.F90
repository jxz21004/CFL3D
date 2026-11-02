! In-memory adapter for the unchanged NASA SPLITTER topology kernels.
! Coordinates and physical BC payloads never enter this routine.
subroutine py_split_topology(dims,segments,owners,peers,block,axis,index,nb,ns,capacity, &
                            result,result_owners,result_peers,bounds,count,status)
  implicit none
  integer,intent(in) :: nb,ns,capacity,block,axis,index
  integer,intent(in) :: dims(3,nb),segments(24,ns),owners(ns),peers(ns)
  integer,intent(out) :: result(24,capacity),result_owners(capacity),result_peers(capacity)
  integer,intent(out) :: bounds(6,nb+1),count,status
  integer :: imap(24,capacity),idbloc(capacity),ivisb(24,nb+1),itrb(7,nb+1)
  double precision :: val(capacity),xdum(24,capacity),xdmold(24)
  integer :: iold(4,capacity),nxtseg(capacity),intrfc(capacity),ipatch(2*capacity+2)
  integer :: nsubbl(nb+1),idobl(nb+1),nseg(nb+1),idno(nb+1),ijk(6,nb+1)
  integer :: idseg(nb+1),idnext(nb+1),idmold(24),b,s,t,nbltop,nsgtop,noldbl
  common /block/ nbltop
  common /segment/ nsgtop
  common /oldbl/ noldbl
  result=0; result_owners=0; result_peers=0; bounds=0; count=0; status=1
  if (nb<1.or.ns<1.or.capacity<8*ns+32) return
  if (block<1.or.block>nb.or.axis<1.or.axis>3) return
  if (index<=1.or.index>=dims(axis,block)) return
  if (any(owners<1).or.any(owners>nb).or.any(peers<0).or.any(peers>ns)) return
  imap=0; idbloc=0; ivisb=0; itrb=0; val=0; xdum=0; iold=0
  nxtseg=0; intrfc=0; ipatch=0; nsubbl=0; idobl=0; nseg=0
  idno=0; ijk=0; idseg=0; idnext=0; xdmold=0; idmold=0
  nbltop=nb; nsgtop=ns; noldbl=nb
  imap(:,1:ns)=segments; idbloc(1:ns)=owners; intrfc(1:ns)=peers
  do b=1,nb
    nsubbl(b)=1; idobl(b)=b; idno(b)=b
    ijk(1:6:2,b)=1; ijk(2:6:2,b)=dims(:,b)
    do s=ns,1,-1
      if (owners(s)/=b) cycle
      nxtseg(s)=idseg(b); idseg(b)=s; nseg(b)=nseg(b)+1
      iold(:,s)=imap(3:6,s)
    end do
    if (nseg(b)==0) return
  end do
  call mapsplt(block,axis,index,nb+1,24,capacity, &
    imap,idbloc,ivisb,itrb,val,xdum,iold,nxtseg,intrfc,ipatch, &
    nsubbl,idobl,nseg,idno,ijk,idseg,idnext,xdmold,idmold)
  ! Equivalent to mapout's reciprocal owner update, without file I/O or renumbering.
  do s=1,nsgtop
    t=intrfc(s)
    if (t>0) imap(7,t)=idbloc(s)
  end do
  count=nsgtop; bounds=ijk
  result(:,1:count)=imap(:,1:count)
  result_owners(1:count)=idbloc(1:count)
  result_peers(1:count)=intrfc(1:count)
  status=0
end subroutine
