module parallelized_matrix_multiply_mod
    use iso_c_binding
    implicit none
contains

    subroutine parallelized_matrix_multiply(A, B, C, n) bind(c, name="parallelized_matrix_multiply")
        use omp_lib
        implicit none
        ! Arguments
        real(c_double), intent(in) :: A(:,:), B(:,:)
        real(c_double), intent(out) :: C(:,:)
        integer(c_int), value :: n
        ! Local variables
        integer :: i, j, k

        ! Initialize result matrix res to zero
        C = 0.0d0
        
        ! OpenMP parallelization: multiply matrices A and B into res
        !$omp parallel do private(i, j, k) shared(A, B, C, n) schedule(dynamic)
        do i = 1, n
            do j = 1, n
                do k = 1, n
                    C(i, j) = C(i, j) + A(i, k) * B(k, j)
                end do
            end do
        end do
        !$omp end parallel do

    end subroutine parallelized_matrix_multiply

end module parallelized_matrix_multiply_mod
