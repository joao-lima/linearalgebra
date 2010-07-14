
#include <iostream>

int main( int argc, char **argv )
{
	lb lbm;
	lbm.read( argv[1], argv[2] );
	lbm.init();
	return 0;
}
