cd /data/local/tmp/snpe
export LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH="$PWD/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp"
# usage: evalrun.sh <dlc> <input dir under eval/>   (list.txt paths are relative to /data/local/tmp/snpe)
rm -rf eval/$2_out
./snpe-net-run --container models/$1 --input_list eval/$2/list.txt --use_dsp --perf_profile burst \
  --use_native_input_files --use_native_output_files --output_dir eval/$2_out > eval/$2.log 2>&1
echo rc=$? results=$(ls eval/$2_out | grep -c Result_); ls eval/$2_out/Result_0
chmod -R a+rwx eval/$2_out
