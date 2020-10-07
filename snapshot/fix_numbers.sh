if [ $# -ne 1 ] ; then
  echo "usage: $0 <filename>"
  exit
fi
cat "$1" | cut -d';' -f2 | awk 'BEGIN{i=0} /#/{print} !/#/{printf "%04d;%s\n",i++,$1}' > "$1.tmp"
mv "$1.tmp" "$1"

