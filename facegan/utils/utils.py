import bz2


def unpack_bz2(src_path):
  data = bz2.BZ2File(src_path).read()
  dst_path = src_path[:-4]
  with open(dst_path, 'wb') as fp:
    fp.write(data)
  return dst_path
