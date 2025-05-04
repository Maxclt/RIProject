module parallelized_sum_powers_mod
    use iso_c_binding
    implicit none
contains

    function parallelized_sum_of_powers(array, n, pow) result(res) bind(c, name="parallelized_sum_of_powers")
        use omp_lib
        implicit none
        ! Arguments
        real(c_double), intent(in) :: array(*)
        integer(c_int), value :: n
        integer(c_int), value :: pow
        ! Local variables
        real(c_double) :: res
        integer :: i

        ! Function body
        res = 0.0d0

        !$omp parallel do reduction(+:res)
        do i = 1, n
            res = res + array(i) ** pow
        end do
        !$omp end parallel do
    end function parallelized_sum_of_powers

end module parallelized_sum_powers_mod