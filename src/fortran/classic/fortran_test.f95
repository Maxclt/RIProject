! fortran_code.f95
module fortran_test_mod
    use iso_c_binding
    implicit none
contains
    function add_numbers(a, b) result(res) bind(C, name="add_numbers")
        real(c_double), value :: a, b
        real(c_double) :: res

        res = a + b
    end function add_numbers
end module fortran_test_mod
