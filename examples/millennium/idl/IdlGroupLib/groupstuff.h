 
int get_spherical_region_count(int argc, void *argv[]);
int get_spherical_region_coordinates(int argc, void *argv[]);



int get_total_number_of_groups(int argc,void *argv[]);
int get_group_catalogue(int argc,void *argv[]);
int get_hash_table(int argc,void *argv[]);
int get_hash_table_size(int argc,void *argv[]);
int get_group_coordinates(int argc,void *argv[]);

int get_minimum_group_len(int argc, void *argv[]);
int get_groupcount_below_minimum_len(int argc, void *argv[]);


int id_sort_compare_key(const void *a, const void *b);
int id_sort_groups(const void *a, const void *b);
double fof_periodic_wrap(double x);
double fof_periodic(double x);


typedef int peanokey;

peanokey peano_hilbert_key(int x, int y, int z, int bits);
void peano_hilbert_key_inverse(peanokey key, int bits, int *x, int *y, int *z);

