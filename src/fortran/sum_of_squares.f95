module sum_squares_mod
    use iso_c_binding
    implicit none
contains

    function sum_of_squares(array, n) result(res) bind(c, name="sum_of_squares")
        implicit none
        ! Arguments
        real(c_double), intent(in) :: array(*)
        integer(c_int), value :: n
        ! Local variables
        real(c_double) :: res
        integer :: i

        ! Function body
        res = 0.0d0
        do i = 1, n
            res = res + array(i) * array(i)
        end do
    end function sum_of_squares

end module sum_squares_mod
