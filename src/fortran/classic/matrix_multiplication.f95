module matrix_multiply_mod
    use iso_c_binding
    implicit none
contains

    subroutine matrix_multiply(A, B, C, n) bind(c, name="matrix_multiply")
        implicit none
        ! Arguments
        real(c_double), intent(in) :: A(:,:), B(:,:)
        real(c_double), intent(out) :: C(:,:)
        integer(c_int), value :: n
        ! Local variables
        integer :: i, j, k

        ! Initialize result matrix C to zero
        C = 0.0d0

        ! Non-parallelized matrix multiplication
        do i = 1, n
            do j = 1, n
                do k = 1, n
                    C(i, j) = C(i, j) + A(i, k) * B(k, j)
                end do
            end do
        end do

    end subroutine matrix_multiply

end module matrix_multiply_mod
