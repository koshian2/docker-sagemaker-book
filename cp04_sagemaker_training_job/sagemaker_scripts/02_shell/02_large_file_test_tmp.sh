cd /tmp
pwd
for i in `seq 100`
do
    dd if=/dev/zero of="dummy${i}" bs=1M count=1024
    echo "Write dummy${i}"
done