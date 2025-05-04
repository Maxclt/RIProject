module sum_powers_mod
    use iso_c_binding
    implicit none
contains

    function sum_of_powers(array, n, pow) result(res) bind(c, name="sum_of_powers")
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

        do i = 1, n
            res = res + array(i) ** pow
        end do
    end function sum_of_powers

end module sum_powers_mod
