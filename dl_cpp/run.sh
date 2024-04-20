# g++ -std=c++11 -o "$1" "$1".cpp -L/opt/homebrew/opt/openblas/lib -I/opt/homebrew/opt/openblas/include -lopenblas

g++ -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -o "$1" "$1".cpp

./"$1"