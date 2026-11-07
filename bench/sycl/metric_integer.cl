/* Independently authored exact-integer collection control. Repository LICENSE.
 * Storage has 64 guard words at either end; dispatch may round up its extent.
 */
__kernel void strata_metric_integer_work(__global unsigned int *data) {
    size_t i = get_global_id(0);
    if (i < 65536) data[64 + i] = data[64 + i] * 3u + 7u;
}
