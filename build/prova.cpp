#include <iostream>
#include <stdint.h>

int main(int argc, char const *argv[]) {
  bool ca,cb,cc;
  std::cout << "ca: "<<&ca << '\n';
  std::cout << "cb: "<<&cb << '\n';
  std::cout << "cc: "<<&cc << '\n';
  short sa,sb,sc;
  std::cout << "sa: "<<&sa << '\n';
  std::cout << "sb: "<<&sb << '\n';
  std::cout << "sc: "<<&sc << '\n';
  int ia,ib,ic;
  std::cout << "ia: "<<&ia << '\n';
  std::cout << "ib: "<<&ib << '\n';
  std::cout << "ic: "<<&ic << '\n';
  double a = 2.323;


  int foo[] = {2,1,3};
  int c = 3;
  //uintptr_t addr = reinterpret_cast<uintptr_t>(&foo);
  //uintptr_t addr2 = reinterpret_cast<uintptr_t>(&c);
  int* prt = &c;
  std::cout << "a: "<<&a << '\n';
  
  std::cout << "foo: "<<&foo  << '\n';
  std::cout << "foo0: "<<&foo[0]  << '\n';
  std::cout << "foo1: "<<&foo[1]  << '\n';
  std::cout << "foo2: "<<&foo[2]  << '\n';
  std::cout << "real &c: "<<&c << '\n';
std::cout << "computed &c: "<<&foo + 0x01 << '\n'; //va avanti di un byte essendo in Hex



  std::cout << "prt: "<<prt << '\n';
  std::cout << "*prt: "<<*prt << '\n';
  return 0;

}
