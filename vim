execve("/home/loganr/.conda/envs/fly/bin/python", ["python", "-c", "from openvino import Core; print"..., "-"], 0x7ffe3b7af878 /* 82 vars */) = 0
brk(NULL)                               = 0x2b332000
readlinkat(AT_FDCWD, "/proc/self/exe", "/home/loganr/.conda/envs/fly/bin"..., 4096) = 43
mmap(NULL, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a8979b000
access("/etc/ld.so.preload", R_OK)      = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/glibc-hwcaps/x86-64-v3/libpthread.so.0", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/glibc-hwcaps/x86-64-v3/", 0x7ffeca376950, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/glibc-hwcaps/x86-64-v2/libpthread.so.0", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/glibc-hwcaps/x86-64-v2/", 0x7ffeca376950, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/libpthread.so.0", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/", {st_mode=S_IFDIR|S_ISGID|0755, st_size=98304, ...}, 0) = 0
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=243639, ...}) = 0
mmap(NULL, 243639, PROT_READ, MAP_PRIVATE, 3, 0) = 0x7f3a8975f000
close(3)                                = 0
openat(AT_FDCWD, "/usr/lib/libpthread.so.0", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=14360, ...}) = 0
mmap(NULL, 16400, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a8975a000
mmap(0x7f3a8975b000, 4096, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1000) = 0x7f3a8975b000
mmap(0x7f3a8975c000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a8975c000
mmap(0x7f3a8975d000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a8975d000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/libdl.so.2", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/usr/lib/libdl.so.2", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=14352, ...}) = 0
mmap(NULL, 16400, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a89755000
mmap(0x7f3a89756000, 4096, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1000) = 0x7f3a89756000
mmap(0x7f3a89757000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a89757000
mmap(0x7f3a89758000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a89758000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/libutil.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/usr/lib/libutil.so.1", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=14352, ...}) = 0
mmap(NULL, 16400, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a89750000
mmap(0x7f3a89751000, 4096, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1000) = 0x7f3a89751000
mmap(0x7f3a89752000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a89752000
mmap(0x7f3a89753000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a89753000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/libm.so.6", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/usr/lib/libm.so.6", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\3\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=1100400, ...}) = 0
mmap(NULL, 1102152, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a89642000
mmap(0x7f3a89651000, 569344, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xf000) = 0x7f3a89651000
mmap(0x7f3a896dc000, 466944, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x9a000) = 0x7f3a896dc000
mmap(0x7f3a8974e000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x10b000) = 0x7f3a8974e000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/libc.so.6", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/usr/lib/libc.so.6", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\3\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\360w\2\0\0\0\0\0"..., 832) = 832
pread64(3, "\6\0\0\0\4\0\0\0@\0\0\0\0\0\0\0@\0\0\0\0\0\0\0@\0\0\0\0\0\0\0"..., 896, 64) = 896
fstat(3, {st_mode=S_IFREG|0755, st_size=2145632, ...}) = 0
pread64(3, "\6\0\0\0\4\0\0\0@\0\0\0\0\0\0\0@\0\0\0\0\0\0\0@\0\0\0\0\0\0\0"..., 896, 64) = 896
mmap(NULL, 2169904, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a89400000
mmap(0x7f3a89424000, 1511424, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x24000) = 0x7f3a89424000
mmap(0x7f3a89595000, 454656, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x195000) = 0x7f3a89595000
mmap(0x7f3a89604000, 24576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x203000) = 0x7f3a89604000
mmap(0x7f3a8960a000, 31792, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a8960a000
close(3)                                = 0
mmap(NULL, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a89640000
arch_prctl(ARCH_SET_FS, 0x7f3a89641500) = 0
set_tid_address(0x7f3a896417d0)         = 1266982
set_robust_list(0x7f3a896417e0, 24)     = 0
rseq(0x7f3a89641440, 0x20, 0, 0x53053053) = 0
mprotect(0x7f3a89604000, 16384, PROT_READ) = 0
mprotect(0x7f3a8974e000, 4096, PROT_READ) = 0
mprotect(0x7f3a89753000, 4096, PROT_READ) = 0
mprotect(0x7f3a89758000, 4096, PROT_READ) = 0
mprotect(0x7f3a8975d000, 4096, PROT_READ) = 0
mprotect(0x722000, 4096, PROT_READ)     = 0
mmap(NULL, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a8963e000
mprotect(0x7f3a897d6000, 8192, PROT_READ) = 0
prlimit64(0, RLIMIT_STACK, NULL, {rlim_cur=8192*1024, rlim_max=RLIM64_INFINITY}) = 0
getrandom("\x67\x35\x6c\xf4\xf7\x7f\xae\xcd", 8, GRND_NONBLOCK) = 8
munmap(0x7f3a8975f000, 243639)          = 0
brk(NULL)                               = 0x2b332000
brk(0x2b353000)                         = 0x2b353000
openat(AT_FDCWD, "/usr/lib/locale/locale-archive", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3063024, ...}) = 0
mmap(NULL, 3063024, PROT_READ, MAP_PRIVATE, 3, 0) = 0x7f3a89000000
close(3)                                = 0
openat(AT_FDCWD, "/usr/lib/gconv/gconv-modules.cache", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=27010, ...}) = 0
mmap(NULL, 27010, PROT_READ, MAP_SHARED, 3, 0) = 0x7f3a89794000
close(3)                                = 0
futex(0x7f3a8960972c, FUTEX_WAKE_PRIVATE, 2147483647) = 0
getrandom("\x10\x77\xe5\x78\xfe\x4c\x6e\x55\xe3\xa4\x35\xf7\xa0\xb6\xd2\xe5\x96\x8c\x45\xa9\xa9\xfd\xd9\x48", 24, GRND_NONBLOCK) = 24
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a89300000
brk(0x2b374000)                         = 0x2b374000
brk(0x2b399000)                         = 0x2b399000
stat("/home/loganr/.conda/envs/fly/bin/python", {st_mode=S_IFREG|0755, st_size=17250880, ...}) = 0
readlink("/home/loganr/.conda/envs/fly/bin/python", "python3.10", 4096) = 10
readlink("/home/loganr/.conda/envs/fly/bin/python3.10", 0x7ffeca371fe0, 4096) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/pyvenv.cfg", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/pyvenv.cfg", O_RDONLY) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/bin/Modules/Setup.local", 0x7ffeca372fb0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/bin/lib/python3.10/os.py", 0x7ffeca372e90) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/bin/lib/python3.10/os.pyc", 0x7ffeca372e90) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/os.py", {st_mode=S_IFREG|0644, st_size=39557, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/pybuilddir.txt", O_RDONLY) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/bin/lib/python3.10/lib-dynload", 0x7ffeca372030) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a88f00000
openat(AT_FDCWD, "/etc/localtime", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2852, ...}) = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2852, ...}) = 0
read(3, "TZif2\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\6\0\0\0\6\0\0\0\0"..., 4096) = 2852
lseek(3, -1810, SEEK_CUR)               = 1042
read(3, "TZif2\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\6\0\0\0\6\0\0\0\0"..., 4096) = 1810
brk(0x2b3ba000)                         = 0x2b3ba000
lseek(3, 2851, SEEK_SET)                = 2851
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python310.zip", 0x7ffeca375970) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=98304, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python310.zip", 0x7ffeca375710) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
getdents64(3, 0x2b39a4a0 /* 210 entries */, 32768) = 7088
getdents64(3, 0x2b39a4a0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/encodings/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca375ae0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/encodings/__init__.abi3.so", 0x7ffeca375ae0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/encodings/__init__.so", 0x7ffeca375ae0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/encodings/__init__.py", {st_mode=S_IFREG|0644, st_size=5620, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/encodings/__init__.py", {st_mode=S_IFREG|0644, st_size=5620, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/encodings/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fcntl(3, F_GETFD)                       = 0x1 (flags FD_CLOEXEC)
fstat(3, {st_mode=S_IFREG|0644, st_size=3870, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375e50)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3870, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\364\25\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3871) = 3870
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/codecs.py", {st_mode=S_IFREG|0644, st_size=36714, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/codecs.py", {st_mode=S_IFREG|0644, st_size=36714, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/codecs.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=33214, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375050)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=33214, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hj\217\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 33215) = 33214
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/encodings", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/encodings", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/encodings", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/encodings", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b3a8580 /* 125 entries */, 32768) = 4224
getdents64(3, 0x2b3a8580 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/encodings/aliases.py", {st_mode=S_IFREG|0644, st_size=15677, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/encodings/aliases.py", {st_mode=S_IFREG|0644, st_size=15677, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/encodings/__pycache__/aliases.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=11175, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3749a0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=11175, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h==\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 11176) = 11175
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/encodings", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/encodings/utf_8.py", {st_mode=S_IFREG|0644, st_size=1005, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/encodings/utf_8.py", {st_mode=S_IFREG|0644, st_size=1005, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/encodings/__pycache__/utf_8.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1851, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375e80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1851, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h\355\3\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1852) = 1851
read(3, "", 1)                          = 0
close(3)                                = 0
rt_sigaction(SIGPIPE, {sa_handler=SIG_IGN, sa_mask=[], sa_flags=SA_RESTORER|SA_ONSTACK, sa_restorer=0x7f3a8943e4d0}, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGXFSZ, {sa_handler=SIG_IGN, sa_mask=[], sa_flags=SA_RESTORER|SA_ONSTACK, sa_restorer=0x7f3a8943e4d0}, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGHUP, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGINT, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGQUIT, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGILL, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGTRAP, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigprocmask(SIG_BLOCK, ~[], [], 8)   = 0
rt_sigaction(SIGABRT, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigprocmask(SIG_SETMASK, [], NULL, 8) = 0
rt_sigaction(SIGBUS, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGFPE, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGKILL, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGUSR1, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGSEGV, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGUSR2, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGPIPE, NULL, {sa_handler=SIG_IGN, sa_mask=[], sa_flags=SA_RESTORER|SA_ONSTACK, sa_restorer=0x7f3a8943e4d0}, 8) = 0
rt_sigaction(SIGALRM, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGTERM, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGSTKFLT, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGCHLD, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGCONT, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGSTOP, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGTSTP, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGTTIN, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGTTOU, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGURG, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGXCPU, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGXFSZ, NULL, {sa_handler=SIG_IGN, sa_mask=[], sa_flags=SA_RESTORER|SA_ONSTACK, sa_restorer=0x7f3a8943e4d0}, 8) = 0
rt_sigaction(SIGVTALRM, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGPROF, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGWINCH, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGIO, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGPWR, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGSYS, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_2, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_3, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_4, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_5, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_6, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_7, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_8, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_9, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_10, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_11, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_12, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_13, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_14, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_15, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_16, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_17, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_18, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_19, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_20, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_21, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_22, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_23, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_24, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_25, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_26, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_27, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_28, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_29, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_30, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_31, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGRT_32, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGINT, {sa_handler=0x495e37, sa_mask=[], sa_flags=SA_RESTORER|SA_ONSTACK, sa_restorer=0x7f3a8943e4d0}, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
fstat(0, {st_mode=S_IFCHR|0620, st_rdev=makedev(0x88, 0xa), ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/io.py", {st_mode=S_IFREG|0644, st_size=4196, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/io.py", {st_mode=S_IFREG|0644, st_size=4196, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/io.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3658, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375f20)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3658, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hd\20\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3659) = 3658
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/abc.py", {st_mode=S_IFREG|0644, st_size=6522, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/abc.py", {st_mode=S_IFREG|0644, st_size=6522, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/abc.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=7005, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375120)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=7005, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\17\272\367hz\31\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 7006) = 7005
read(3, "", 1)                          = 0
close(3)                                = 0
dup(0)                                  = 3
close(3)                                = 0
fstat(0, {st_mode=S_IFCHR|0620, st_rdev=makedev(0x88, 0xa), ...}) = 0
ioctl(0, TCGETS2, {c_iflag=ICRNL|IXON|IUTF8, c_oflag=NL0|CR0|TAB0|BS0|VT0|FF0|OPOST|ONLCR, c_cflag=B38400|B38400<<IBSHIFT|CS8|CREAD, c_lflag=ISIG|ICANON|ECHO|ECHOE|ECHOK|IEXTEN|ECHOCTL|ECHOKE, ...}) = 0
lseek(0, 0, SEEK_CUR)                   = -1 ESPIPE (Illegal seek)
ioctl(0, TCGETS2, {c_iflag=ICRNL|IXON|IUTF8, c_oflag=NL0|CR0|TAB0|BS0|VT0|FF0|OPOST|ONLCR, c_cflag=B38400|B38400<<IBSHIFT|CS8|CREAD, c_lflag=ISIG|ICANON|ECHO|ECHOE|ECHOK|IEXTEN|ECHOCTL|ECHOKE, ...}) = 0
dup(1)                                  = 3
close(3)                                = 0
fstat(1, {st_mode=S_IFCHR|0620, st_rdev=makedev(0x88, 0xa), ...}) = 0
ioctl(1, TCGETS2, {c_iflag=ICRNL|IXON|IUTF8, c_oflag=NL0|CR0|TAB0|BS0|VT0|FF0|OPOST|ONLCR, c_cflag=B38400|B38400<<IBSHIFT|CS8|CREAD, c_lflag=ISIG|ICANON|ECHO|ECHOE|ECHOK|IEXTEN|ECHOCTL|ECHOKE, ...}) = 0
lseek(1, 0, SEEK_CUR)                   = -1 ESPIPE (Illegal seek)
ioctl(1, TCGETS2, {c_iflag=ICRNL|IXON|IUTF8, c_oflag=NL0|CR0|TAB0|BS0|VT0|FF0|OPOST|ONLCR, c_cflag=B38400|B38400<<IBSHIFT|CS8|CREAD, c_lflag=ISIG|ICANON|ECHO|ECHOE|ECHOK|IEXTEN|ECHOCTL|ECHOKE, ...}) = 0
dup(2)                                  = 3
close(3)                                = 0
fstat(2, {st_mode=S_IFREG|0644, st_size=24267, ...}) = 0
ioctl(2, TCGETS2, 0x7ffeca376ec0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(2, 0, SEEK_CUR)                   = 24422
ioctl(2, TCGETS2, 0x7ffeca377180)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(2, 0, SEEK_CUR)                   = 24555
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site.py", {st_mode=S_IFREG|0644, st_size=22926, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site.py", {st_mode=S_IFREG|0644, st_size=22926, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/site.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=17902, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375f20)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=17902, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\216Y\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 17903) = 17902
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/os.py", {st_mode=S_IFREG|0644, st_size=39557, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/os.py", {st_mode=S_IFREG|0644, st_size=39557, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/os.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=31594, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375120)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=31594, ...}) = 0
brk(0x2b3de000)                         = 0x2b3de000
read(3, "o\r\r\n\0\0\0\0\20\272\367h\205\232\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 31595) = 31594
read(3, "", 1)                          = 0
close(3)                                = 0
mmap(NULL, 151552, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a8976f000
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/stat.py", {st_mode=S_IFREG|0644, st_size=5485, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/stat.py", {st_mode=S_IFREG|0644, st_size=5485, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/stat.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4527, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374320)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4527, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hm\25\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4528) = 4527
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/_collections_abc.py", {st_mode=S_IFREG|0644, st_size=32284, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/_collections_abc.py", {st_mode=S_IFREG|0644, st_size=32284, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/_collections_abc.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=32920, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374320)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=32920, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\17\272\367h\34~\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 32921) = 32920
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/posixpath.py", {st_mode=S_IFREG|0644, st_size=16436, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/posixpath.py", {st_mode=S_IFREG|0644, st_size=16436, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/posixpath.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=10646, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374320)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=10646, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h4@\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 10647) = 10646
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/genericpath.py", {st_mode=S_IFREG|0644, st_size=5246, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/genericpath.py", {st_mode=S_IFREG|0644, st_size=5246, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/genericpath.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4680, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373520)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4680, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h~\24\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4681) = 4680
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/_sitebuiltins.py", {st_mode=S_IFREG|0644, st_size=3128, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/_sitebuiltins.py", {st_mode=S_IFREG|0644, st_size=3128, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/_sitebuiltins.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3801, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375120)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3801, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\17\272\367h8\f\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3802) = 3801
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/bin/pyvenv.cfg", 0x7ffeca375980) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/pyvenv.cfg", 0x7ffeca375980) = -1 ENOENT (No such file or directory)
geteuid()                               = 1000
getuid()                                = 1000
getegid()                               = 1000
getgid()                                = 1000
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.local/lib/python3.10/site-packages", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b3bf280 /* 2 entries */, 32768) = 48
getdents64(3, 0x2b3bf280 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
getdents64(3, 0x2b3bf280 /* 371 entries */, 32768) = 15120
getdents64(3, 0x2b3bf280 /* 0 entries */, 32768) = 0
close(3)                                = 0
lstat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/distutils-precedence.pth", {st_mode=S_IFREG|0644, st_size=151, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/distutils-precedence.pth", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=151, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375710)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
ioctl(3, TCGETS2, 0x7ffeca3759e0)       = -1 ENOTTY (Inappropriate ioctl for device)
read(3, "import os; var = 'SETUPTOOLS_USE"..., 8192) = 151
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 4
fstat(4, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(4, 0x2b3c12b0 /* 77 entries */, 32768) = 4840
getdents64(4, 0x2b3c12b0 /* 0 entries */, 32768) = 0
close(4)                                = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.local/lib/python3.10/site-packages", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 4
fstat(4, {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getdents64(4, 0x2b3c12b0 /* 2 entries */, 32768) = 48
getdents64(4, 0x2b3c12b0 /* 0 entries */, 32768) = 0
close(4)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 4
fstat(4, {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
getdents64(4, 0x2b3c12b0 /* 371 entries */, 32768) = 15120
getdents64(4, 0x2b3c12b0 /* 0 entries */, 32768) = 0
close(4)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/_distutils_hack/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374280) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/_distutils_hack/__init__.abi3.so", 0x7ffeca374280) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/_distutils_hack/__init__.so", 0x7ffeca374280) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/_distutils_hack/__init__.py", {st_mode=S_IFREG|0644, st_size=6755, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/_distutils_hack/__init__.py", {st_mode=S_IFREG|0644, st_size=6755, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/_distutils_hack/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 4
fstat(4, {st_mode=S_IFREG|0644, st_size=8206, ...}) = 0
ioctl(4, TCGETS2, 0x7ffeca3745f0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(4, 0, SEEK_CUR)                   = 0
lseek(4, 0, SEEK_CUR)                   = 0
fstat(4, {st_mode=S_IFREG|0644, st_size=8206, ...}) = 0
read(4, "o\r\r\n\0\0\0\0\277\r5hc\32\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 8207) = 8206
read(4, "", 1)                          = 0
close(4)                                = 0
read(3, "", 8192)                       = 0
close(3)                                = 0
lstat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/flygym.pth", {st_mode=S_IFREG|0644, st_size=52, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/flygym.pth", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=52, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375710)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
ioctl(3, TCGETS2, 0x7ffeca3759e0)       = -1 ENOTTY (Inappropriate ioctl for device)
read(3, "/home/loganr/Desktop/programming"..., 8192) = 52
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
read(3, "", 8192)                       = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/Desktop/programming/fly/project/flygym", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b3cb2d0 /* 19 entries */, 32768) = 616
getdents64(3, 0x2b3cb2d0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/Desktop/programming/fly/project", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b3cb2d0 /* 36 entries */, 32768) = 1192
getdents64(3, 0x2b3cb2d0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca375ec0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__init__.abi3.so", 0x7ffeca375ec0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__init__.so", 0x7ffeca375ec0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__init__.py", {st_mode=S_IFREG|0644, st_size=3547, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__init__.py", {st_mode=S_IFREG|0644, st_size=3547, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2729, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca376230)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2729, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\333\r\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2730) = 2729
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/pkgutil.py", {st_mode=S_IFREG|0644, st_size=24576, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/pkgutil.py", {st_mode=S_IFREG|0644, st_size=24576, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/pkgutil.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=18356, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375380)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=18356, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\0`\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 18357) = 18356
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/collections/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/collections/__init__.abi3.so", 0x7ffeca374210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/collections/__init__.so", 0x7ffeca374210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/collections/__init__.py", {st_mode=S_IFREG|0644, st_size=51398, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/collections/__init__.py", {st_mode=S_IFREG|0644, st_size=51398, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/collections/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=48707, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=48707, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\306\310\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 48708) = 48707
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/keyword.py", {st_mode=S_IFREG|0644, st_size=1061, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/keyword.py", {st_mode=S_IFREG|0644, st_size=1061, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/keyword.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=922, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=922, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h%\4\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 923) = 922
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/operator.py", {st_mode=S_IFREG|0644, st_size=10751, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/operator.py", {st_mode=S_IFREG|0644, st_size=10751, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/operator.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=13503, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=13503, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\377)\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 13504) = 13503
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/reprlib.py", {st_mode=S_IFREG|0644, st_size=5267, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/reprlib.py", {st_mode=S_IFREG|0644, st_size=5267, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/reprlib.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5504, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5504, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\223\24\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5505) = 5504
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/functools.py", {st_mode=S_IFREG|0644, st_size=38076, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/functools.py", {st_mode=S_IFREG|0644, st_size=38076, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/functools.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=28330, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=28330, ...}) = 0
brk(0x2b3ff000)                         = 0x2b3ff000
read(3, "o\r\r\n\0\0\0\0\20\272\367h\274\224\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 28331) = 28330
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2b3f7000)                         = 0x2b3f7000
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/types.py", {st_mode=S_IFREG|0644, st_size=10117, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/types.py", {st_mode=S_IFREG|0644, st_size=10117, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/types.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=9520, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=9520, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\205'\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 9521) = 9520
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/__init__.abi3.so", 0x7ffeca374210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/__init__.so", 0x7ffeca374210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/__init__.py", {st_mode=S_IFREG|0644, st_size=6089, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/__init__.py", {st_mode=S_IFREG|0644, st_size=6089, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4059, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4059, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h\311\27\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4060) = 4059
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/warnings.py", {st_mode=S_IFREG|0644, st_size=19688, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/warnings.py", {st_mode=S_IFREG|0644, st_size=19688, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/warnings.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=13900, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=13900, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\350L\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 13901) = 13900
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b3de6f0 /* 15 entries */, 32768) = 488
getdents64(3, 0x2b3de6f0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/util.py", {st_mode=S_IFREG|0644, st_size=11487, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/util.py", {st_mode=S_IFREG|0644, st_size=11487, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib/__pycache__/util.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=9583, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=9583, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h\337,\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 9584) = 9583
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/_abc.py", {st_mode=S_IFREG|0644, st_size=1852, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/_abc.py", {st_mode=S_IFREG|0644, st_size=1852, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib/__pycache__/_abc.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2225, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2225, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h<\7\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2226) = 2225
read(3, "", 1)                          = 0
close(3)                                = 0
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a88e00000
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/contextlib.py", {st_mode=S_IFREG|0644, st_size=25882, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/contextlib.py", {st_mode=S_IFREG|0644, st_size=25882, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/contextlib.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=21149, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=21149, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\32e\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 21150) = 21149
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/machinery.py", {st_mode=S_IFREG|0644, st_size=831, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/machinery.py", {st_mode=S_IFREG|0644, st_size=831, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib/__pycache__/machinery.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=939, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=939, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h?\3\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 940) = 939
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/weakref.py", {st_mode=S_IFREG|0644, st_size=21560, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/weakref.py", {st_mode=S_IFREG|0644, st_size=21560, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/weakref.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=20338, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3743c0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=20338, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h8T\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 20339) = 20338
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/_weakrefset.py", {st_mode=S_IFREG|0644, st_size=5923, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/_weakrefset.py", {st_mode=S_IFREG|0644, st_size=5923, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/_weakrefset.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=7603, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3735c0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=7603, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\17\272\367h#\27\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 7604) = 7603
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project/.", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/Desktop/programming/fly/project/.", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b3ea3d0 /* 36 entries */, 32768) = 1192
getdents64(3, 0x2b3ea3d0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("openvino.pkg", 0x7ffeca376010)    = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python310.zip/openvino.pkg", 0x7ffeca376010) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/openvino.pkg", 0x7ffeca376010) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/openvino.pkg", 0x7ffeca376010) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages/openvino.pkg", 0x7ffeca376010) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca375ad0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__init__.abi3.so", 0x7ffeca375ad0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__init__.so", 0x7ffeca375ad0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__init__.py", {st_mode=S_IFREG|0644, st_size=3547, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino.pkg", 0x7ffeca376010) = -1 ENOENT (No such file or directory)
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym/openvino.pkg", 0x7ffeca376010) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b3eabe0 /* 46 entries */, 32768) = 1520
getdents64(3, 0x2b3eabe0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/package_utils.py", {st_mode=S_IFREG|0644, st_size=5973, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/package_utils.py", {st_mode=S_IFREG|0644, st_size=5973, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__pycache__/package_utils.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5951, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375430)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5951, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215iU\27\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5952) = 5951
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/typing.py", {st_mode=S_IFREG|0644, st_size=92557, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/typing.py", {st_mode=S_IFREG|0644, st_size=92557, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/typing.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=85531, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374630)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=85531, ...}) = 0
brk(0x2b422000)                         = 0x2b422000
read(3, "o\r\r\n\0\0\0\0\20\272\367h\215i\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 85532) = 85531
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/collections", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/collections", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/collections", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/collections", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b3ec970 /* 5 entries */, 32768) = 144
getdents64(3, 0x2b3ec970 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/collections/abc.py", {st_mode=S_IFREG|0644, st_size=119, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/collections/abc.py", {st_mode=S_IFREG|0644, st_size=119, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/collections/__pycache__/abc.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=233, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373830)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=233, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hw\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 234) = 233
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/re.py", {st_mode=S_IFREG|0644, st_size=15860, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/re.py", {st_mode=S_IFREG|0644, st_size=15860, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/re.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=14481, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373830)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=14481, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\364=\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 14482) = 14481
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/enum.py", {st_mode=S_IFREG|0644, st_size=39831, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/enum.py", {st_mode=S_IFREG|0644, st_size=39831, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/enum.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=26317, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=26317, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\227\233\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 26318) = 26317
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/sre_compile.py", {st_mode=S_IFREG|0644, st_size=27973, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/sre_compile.py", {st_mode=S_IFREG|0644, st_size=27973, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/sre_compile.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=15448, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=15448, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hEm\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 15449) = 15448
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/sre_parse.py", {st_mode=S_IFREG|0644, st_size=40779, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/sre_parse.py", {st_mode=S_IFREG|0644, st_size=40779, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/sre_parse.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=21750, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=21750, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hK\237\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 21751) = 21750
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/sre_constants.py", {st_mode=S_IFREG|0644, st_size=7177, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/sre_constants.py", {st_mode=S_IFREG|0644, st_size=7177, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/sre_constants.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6352, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370e30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6352, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\t\34\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6353) = 6352
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/copyreg.py", {st_mode=S_IFREG|0644, st_size=7426, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/copyreg.py", {st_mode=S_IFREG|0644, st_size=7426, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/copyreg.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4678, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4678, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\2\35\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4679) = 4678
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/pathlib.py", {st_mode=S_IFREG|0644, st_size=49575, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/pathlib.py", {st_mode=S_IFREG|0644, st_size=49575, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/pathlib.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=42047, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374630)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=42047, ...}) = 0
brk(0x2b445000)                         = 0x2b445000
read(3, "o\r\r\n\0\0\0\0\20\272\367h\247\301\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 42048) = 42047
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2b43b000)                         = 0x2b43b000
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/fnmatch.py", {st_mode=S_IFREG|0644, st_size=6713, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/fnmatch.py", {st_mode=S_IFREG|0644, st_size=6713, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/fnmatch.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4498, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373830)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4498, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h9\32\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4499) = 4498
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ntpath.py", {st_mode=S_IFREG|0644, st_size=29944, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ntpath.py", {st_mode=S_IFREG|0644, st_size=29944, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/ntpath.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=15536, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373830)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=15536, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\370t\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 15537) = 15536
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca372c50) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib/__init__.abi3.so", 0x7ffeca372c50) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib/__init__.so", 0x7ffeca372c50) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib/__init__.py", {st_mode=S_IFREG|0644, st_size=0, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib/__init__.py", {st_mode=S_IFREG|0644, st_size=0, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/urllib/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=126, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372fc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=126, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h\0\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 127) = 126
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/urllib", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b41eaf0 /* 9 entries */, 32768) = 280
getdents64(3, 0x2b41eaf0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib/parse.py", {st_mode=S_IFREG|0644, st_size=44402, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib/parse.py", {st_mode=S_IFREG|0644, st_size=44402, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/urllib/__pycache__/parse.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=34863, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373830)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=34863, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367hr\255\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 34864) = 34863
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ipaddress.py", {st_mode=S_IFREG|0644, st_size=80837, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ipaddress.py", {st_mode=S_IFREG|0644, st_size=80837, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/ipaddress.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=64509, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=64509, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\305;\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 64510) = 64509
read(3, "", 1)                          = 0
close(3)                                = 0
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a88d00000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/_pyopenvino/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca3750c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/_pyopenvino/__init__.abi3.so", 0x7ffeca3750c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/_pyopenvino/__init__.so", 0x7ffeca3750c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/_pyopenvino/__init__.py", 0x7ffeca3750c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/_pyopenvino/__init__.pyc", 0x7ffeca3750c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/_pyopenvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/_pyopenvino.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0644, st_size=5321641, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/_pyopenvino.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\220\240\6\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0644, st_size=5321641, ...}) = 0
mmap(NULL, 5338024, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a88600000
mprotect(0x7f3a88666000, 4792320, PROT_NONE) = 0
mmap(0x7f3a88666000, 3620864, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x66000) = 0x7f3a88666000
mmap(0x7f3a889da000, 1056768, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3da000) = 0x7f3a889da000
mmap(0x7f3a88add000, 98304, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4dc000) = 0x7f3a88add000
mmap(0x7f3a88af5000, 8360, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a88af5000
mmap(0x7f3a88af8000, 131072, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4f4000) = 0x7f3a88af8000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/glibc-hwcaps/x86-64-v3/libopenvino.so.2540", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/glibc-hwcaps/x86-64-v3/", 0x7ffeca374270, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/glibc-hwcaps/x86-64-v2/libopenvino.so.2540", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/glibc-hwcaps/x86-64-v2/", 0x7ffeca374270, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libopenvino.so.2540", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\320\310\36\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0644, st_size=18064633, ...}) = 0
mmap(NULL, 18171128, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a87400000
mprotect(0x7f3a875e1000, 15425536, PROT_NONE) = 0
mmap(0x7f3a875e1000, 12324864, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1e1000) = 0x7f3a875e1000
mmap(0x7f3a881a2000, 2658304, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xda2000) = 0x7f3a881a2000
mmap(0x7f3a8842c000, 335872, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x102b000) = 0x7f3a8842c000
mmap(0x7f3a8847e000, 101864, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a8847e000
mmap(0x7f3a88497000, 778240, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x107d000) = 0x7f3a88497000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libtbb.so.12", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\20\332\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0644, st_size=367649, ...}) = 0
mmap(NULL, 461856, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a88c8f000
mprotect(0x7f3a88c9c000, 397312, PROT_NONE) = 0
mmap(0x7f3a88c9c000, 217088, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xd000) = 0x7f3a88c9c000
mmap(0x7f3a88cd1000, 69632, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x42000) = 0x7f3a88cd1000
mmap(0x7f3a88ce2000, 20480, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x52000) = 0x7f3a88ce2000
mmap(0x7f3a88ce7000, 87280, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a88ce7000
mmap(0x7f3a88cfd000, 12288, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x57000) = 0x7f3a88cfd000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libstdc++.so.6", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/libstdc++.so.6", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\3\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=14144696, ...}) = 0
mmap(NULL, 2017416, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a87213000
mmap(0x7f3a872c1000, 659456, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xae000) = 0x7f3a872c1000
mmap(0x7f3a87362000, 557056, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x14f000) = 0x7f3a87362000
mmap(0x7f3a873ea000, 73728, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1d7000) = 0x7f3a873ea000
mmap(0x7f3a873fc000, 14472, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a873fc000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libgcc_s.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/libgcc_s.so.1", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0644, st_size=653704, ...}) = 0
mmap(NULL, 124200, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a8961f000
mmap(0x7f3a89623000, 86016, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a89623000
mmap(0x7f3a89638000, 16384, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x19000) = 0x7f3a89638000
mmap(0x7f3a8963c000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1c000) = 0x7f3a8963c000
close(3)                                = 0
mprotect(0x7f3a8963c000, 4096, PROT_READ) = 0
brk(0x2b45c000)                         = 0x2b45c000
mprotect(0x7f3a873ea000, 57344, PROT_READ) = 0
mprotect(0x7f3a88ce2000, 8192, PROT_READ) = 0
mprotect(0x7f3a8842c000, 319488, PROT_READ) = 0
mprotect(0x7f3a88add000, 94208, PROT_READ) = 0
futex(0x7f3a873fcedc, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a873fcee8, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a88cfb2b0, FUTEX_WAKE_PRIVATE, 2147483647) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libtbbmalloc.so.2", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\200c\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0644, st_size=158129, ...}) = 0
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libtbbmalloc.so.2", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\200c\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0644, st_size=158129, ...}) = 0
mmap(NULL, 293296, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a88c47000
mprotect(0x7f3a88c4d000, 266240, PROT_NONE) = 0
mmap(0x7f3a88c4d000, 90112, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x6000) = 0x7f3a88c4d000
mmap(0x7f3a88c63000, 24576, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1c000) = 0x7f3a88c63000
mmap(0x7f3a88c6a000, 12288, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x22000) = 0x7f3a88c6a000
mmap(0x7f3a88c6d000, 131792, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a88c6d000
mmap(0x7f3a88c8e000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x26000) = 0x7f3a88c8e000
close(3)                                = 0
mprotect(0x7f3a88c6a000, 4096, PROT_READ) = 0
futex(0x7f3a88cfa250, FUTEX_WAKE_PRIVATE, 2147483647) = 0
mmap(NULL, 327680, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a88bf7000
openat(AT_FDCWD, "/proc/meminfo", O_RDONLY) = 3
fstat(3, {st_mode=S_IFREG|0444, st_size=0, ...}) = 0
read(3, "MemTotal:       65551164 kB\nMemF"..., 1024) = 1024
read(3, "     171196 kB\nVmallocChunk:    "..., 1024) = 563
lseek(3, 0, SEEK_CUR)                   = 1587
lseek(3, 1475, SEEK_SET)                = 1475
close(3)                                = 0
openat(AT_FDCWD, "/proc/sys/vm/nr_hugepages", O_RDONLY) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=0, ...}) = 0
read(3, "0\n", 1024)                    = 2
close(3)                                = 0
openat(AT_FDCWD, "/sys/kernel/mm/transparent_hugepage/enabled", O_RDONLY) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4096, ...}) = 0
read(3, "[always] madvise never\n", 4096) = 23
close(3)                                = 0
mmap(NULL, 2097152, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a87013000
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libtcm.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libtcm.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/libtcm.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=243639, ...}) = 0
mmap(NULL, 243639, PROT_READ, MAP_PRIVATE, 3, 0) = 0x7f3a88bbb000
close(3)                                = 0
openat(AT_FDCWD, "/usr/lib/glibc-hwcaps/x86-64-v3/libtcm.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/usr/lib/glibc-hwcaps/x86-64-v3/", 0x7ffeca374130, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/usr/lib/glibc-hwcaps/x86-64-v2/libtcm.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/usr/lib/glibc-hwcaps/x86-64-v2/", 0x7ffeca374130, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/usr/lib/libtcm.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/usr/lib/", {st_mode=S_IFDIR|0755, st_size=299008, ...}, 0) = 0
munmap(0x7f3a88bbb000, 243639)          = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libtcm.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
brk(0x2b47d000)                         = 0x2b47d000
brk(0x2b49e000)                         = 0x2b49e000
brk(0x2b4bf000)                         = 0x2b4bf000
brk(0x2b4e0000)                         = 0x2b4e0000
brk(0x2b501000)                         = 0x2b501000
brk(0x2b525000)                         = 0x2b525000
brk(0x2b51c000)                         = 0x2b51c000
mmap(NULL, 299008, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a88bae000
munmap(0x7f3a8976f000, 151552)          = 0
brk(0x2b53d000)                         = 0x2b53d000
brk(0x2b55e000)                         = 0x2b55e000
brk(0x2b57f000)                         = 0x2b57f000
brk(0x2b5a0000)                         = 0x2b5a0000
brk(0x2b5c1000)                         = 0x2b5c1000
brk(0x2b5e2000)                         = 0x2b5e2000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/_op_base.py", {st_mode=S_IFREG|0644, st_size=832, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/_op_base.py", {st_mode=S_IFREG|0644, st_size=832, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__pycache__/_op_base.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1015, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375430)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1015, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i@\3\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1016) = 1015
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/_ov_api.py", {st_mode=S_IFREG|0644, st_size=30240, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/_ov_api.py", {st_mode=S_IFREG|0644, st_size=30240, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__pycache__/_ov_api.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=30066, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375430)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=30066, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i v\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 30067) = 30066
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/traceback.py", {st_mode=S_IFREG|0644, st_size=26222, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/traceback.py", {st_mode=S_IFREG|0644, st_size=26222, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/traceback.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=21707, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374630)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=21707, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hnf\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 21708) = 21707
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/linecache.py", {st_mode=S_IFREG|0644, st_size=5690, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/linecache.py", {st_mode=S_IFREG|0644, st_size=5690, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/linecache.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4396, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373830)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4396, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h:\26\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4397) = 4396
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/tokenize.py", {st_mode=S_IFREG|0644, st_size=25921, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/tokenize.py", {st_mode=S_IFREG|0644, st_size=25921, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/tokenize.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=17448, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=17448, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hAe\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 17449) = 17448
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/token.py", {st_mode=S_IFREG|0644, st_size=2386, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/token.py", {st_mode=S_IFREG|0644, st_size=2386, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/token.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2992, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2992, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hR\t\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2993) = 2992
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca373a50) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/__init__.abi3.so", 0x7ffeca373a50) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/__init__.so", 0x7ffeca373a50) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/__init__.py", {st_mode=S_IFREG|0644, st_size=547, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/__init__.py", {st_mode=S_IFREG|0644, st_size=547, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=637, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373dc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=637, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i#\2\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 638) = 637
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b5d7290 /* 18 entries */, 32768) = 648
getdents64(3, 0x2b5d7290 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/postponed_constant.py", {st_mode=S_IFREG|0644, st_size=3581, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/postponed_constant.py", {st_mode=S_IFREG|0644, st_size=3581, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/__pycache__/postponed_constant.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3981, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372fc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3981, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\375\r\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3982) = 3981
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca3742c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers/__init__.abi3.so", 0x7ffeca3742c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers/__init__.so", 0x7ffeca3742c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers/__init__.py", {st_mode=S_IFREG|0644, st_size=370, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers/__init__.py", {st_mode=S_IFREG|0644, st_size=370, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=421, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374630)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=421, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215ir\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 422) = 421
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b5d7a20 /* 9 entries */, 32768) = 288
getdents64(3, 0x2b5d7a20 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers/data_dispatcher.py", {st_mode=S_IFREG|0644, st_size=15159, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers/data_dispatcher.py", {st_mode=S_IFREG|0644, st_size=15159, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers/__pycache__/data_dispatcher.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=9702, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373830)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=9702, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i7;\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 9703) = 9702
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca3726c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__init__.abi3.so", 0x7ffeca3726c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__init__.so", 0x7ffeca3726c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__init__.py", {st_mode=S_IFREG|0644, st_size=17005, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__init__.py", {st_mode=S_IFREG|0644, st_size=17005, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=11547, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=11547, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302emB\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 11548) = 11547
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
brk(0x2b605000)                         = 0x2b605000
getdents64(3, 0x2b5dcdd0 /* 42 entries */, 32768) = 1352
getdents64(3, 0x2b5dcdd0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_globals.py", {st_mode=S_IFREG|0644, st_size=3094, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_globals.py", {st_mode=S_IFREG|0644, st_size=3094, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__pycache__/_globals.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3457, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3457, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\26\f\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3458) = 3457
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca370ac0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils/__init__.abi3.so", 0x7ffeca370ac0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils/__init__.so", 0x7ffeca370ac0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils/__init__.py", {st_mode=S_IFREG|0644, st_size=723, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils/__init__.py", {st_mode=S_IFREG|0644, st_size=723, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1006, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370e30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1006, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\323\2\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1007) = 1006
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b5dcdd0 /* 7 entries */, 32768) = 216
getdents64(3, 0x2b5dcdd0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils/_convertions.py", {st_mode=S_IFREG|0644, st_size=329, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils/_convertions.py", {st_mode=S_IFREG|0644, st_size=329, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils/__pycache__/_convertions.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=588, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370030)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=588, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302eI\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 589) = 588
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/exceptions.py", {st_mode=S_IFREG|0644, st_size=7339, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/exceptions.py", {st_mode=S_IFREG|0644, st_size=7339, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__pycache__/exceptions.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=7651, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=7651, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\253\34\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 7652) = 7651
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/version.py", {st_mode=S_IFREG|0644, st_size=216, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/version.py", {st_mode=S_IFREG|0644, st_size=216, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__pycache__/version.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=335, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=335, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\330\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 336) = 335
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_distributor_init.py", {st_mode=S_IFREG|0644, st_size=407, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_distributor_init.py", {st_mode=S_IFREG|0644, st_size=407, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__pycache__/_distributor_init.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=584, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=584, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\227\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 585) = 584
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__config__.py", {st_mode=S_IFREG|0644, st_size=7422, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__config__.py", {st_mode=S_IFREG|0644, st_size=7422, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__pycache__/__config__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6496, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6496, ...}) = 0
read(3, "o\r\r\n\0\0\0\0xp\204i\376\34\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6497) = 6496
read(3, "", 1)                          = 0
close(3)                                = 0
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a86f13000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca370250) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__init__.abi3.so", 0x7ffeca370250) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__init__.so", 0x7ffeca370250) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__init__.py", {st_mode=S_IFREG|0644, st_size=5780, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__init__.py", {st_mode=S_IFREG|0644, st_size=5780, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3977, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3705c0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3977, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\224\26\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3978) = 3977
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b5e0640 /* 59 entries */, 32768) = 2360
getdents64(3, 0x2b5e0640 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/multiarray.py", {st_mode=S_IFREG|0644, st_size=56097, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/multiarray.py", {st_mode=S_IFREG|0644, st_size=56097, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/multiarray.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=54337, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=54337, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e!\333\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 54338) = 54337
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/overrides.py", {st_mode=S_IFREG|0644, st_size=7093, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/overrides.py", {st_mode=S_IFREG|0644, st_size=7093, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/overrides.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6170, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36dc60)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6170, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\265\33\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6171) = 6170
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils/_inspect.py", {st_mode=S_IFREG|0644, st_size=7447, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils/_inspect.py", {st_mode=S_IFREG|0644, st_size=7447, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_utils/__pycache__/_inspect.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=7587, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36ce60)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=7587, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\27\35\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 7588) = 7587
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_multiarray_umath.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=6418176, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_multiarray_umath.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=6418176, ...}) = 0
mmap(NULL, 6150608, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a86800000
mmap(0x7f3a8682c000, 4268032, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2c000) = 0x7f3a8682c000
mmap(0x7f3a86c3e000, 1417216, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x43e000) = 0x7f3a86c3e000
mmap(0x7f3a86d98000, 155648, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x598000) = 0x7f3a86d98000
mmap(0x7f3a86dbe000, 129488, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a86dbe000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../../glibc-hwcaps/x86-64-v3/libcblas.so.3", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../../glibc-hwcaps/x86-64-v3/", 0x7ffeca36bca0, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../../glibc-hwcaps/x86-64-v2/libcblas.so.3", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../../glibc-hwcaps/x86-64-v2/", 0x7ffeca36bca0, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../../libcblas.so.3", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0@`\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=131200, ...}) = 0
mmap(NULL, 114720, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a89777000
mmap(0x7f3a8977d000, 69632, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x6000) = 0x7f3a8977d000
mmap(0x7f3a8978e000, 16384, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x17000) = 0x7f3a8978e000
mmap(0x7f3a89792000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1a000) = 0x7f3a89792000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../.././glibc-hwcaps/x86-64-v3/libblas.so.3", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../.././glibc-hwcaps/x86-64-v3/", 0x7ffeca36bc20, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../.././glibc-hwcaps/x86-64-v2/libblas.so.3", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../.././glibc-hwcaps/x86-64-v2/", 0x7ffeca36bc20, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../.././libblas.so.3", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0@0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=476536, ...}) = 0
mmap(NULL, 466968, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a88b3b000
mmap(0x7f3a88b3e000, 434176, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a88b3e000
mmap(0x7f3a88ba8000, 16384, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x6d000) = 0x7f3a88ba8000
mmap(0x7f3a88bac000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x70000) = 0x7f3a88bac000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../.././libgfortran.so.5", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=9106320, ...}) = 0
mmap(NULL, 2807808, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a86400000
mprotect(0x7f3a8641f000, 2670592, PROT_NONE) = 0
mmap(0x7f3a8641f000, 2461696, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1f000) = 0x7f3a8641f000
mmap(0x7f3a86678000, 204800, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x278000) = 0x7f3a86678000
mmap(0x7f3a866ab000, 12288, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2aa000) = 0x7f3a866ab000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../.././libgomp.so.1", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=1462104, ...}) = 0
mmap(NULL, 1102736, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a86e05000
mmap(0x7f3a86e27000, 729088, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x22000) = 0x7f3a86e27000
mmap(0x7f3a86ed9000, 167936, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xd4000) = 0x7f3a86ed9000
mmap(0x7f3a86f02000, 28672, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xfd000) = 0x7f3a86f02000
mmap(0x7f3a86f09000, 37776, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a86f09000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../.././libquadmath.so.0", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=767936, ...}) = 0
mmap(NULL, 225928, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a885c8000
mmap(0x7f3a885cb000, 118784, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a885cb000
mmap(0x7f3a885e8000, 90112, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x20000) = 0x7f3a885e8000
mmap(0x7f3a885fe000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x36000) = 0x7f3a885fe000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../../././glibc-hwcaps/x86-64-v3/librt.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../../././glibc-hwcaps/x86-64-v3/", 0x7ffeca36bb60, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../../././glibc-hwcaps/x86-64-v2/librt.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../../././glibc-hwcaps/x86-64-v2/", 0x7ffeca36bb60, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../../././librt.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../../././", {st_mode=S_IFDIR|S_ISGID|0755, st_size=98304, ...}, 0) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../.././librt.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/../../../../librt.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/librt.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=243639, ...}) = 0
mmap(NULL, 243639, PROT_READ, MAP_PRIVATE, 3, 0) = 0x7f3a8858c000
close(3)                                = 0
openat(AT_FDCWD, "/usr/lib/librt.so.1", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=14424, ...}) = 0
mmap(NULL, 16400, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a89772000
mmap(0x7f3a89773000, 4096, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1000) = 0x7f3a89773000
mmap(0x7f3a89774000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a89774000
mmap(0x7f3a89775000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a89775000
close(3)                                = 0
mprotect(0x7f3a89775000, 4096, PROT_READ) = 0
mprotect(0x7f3a885fe000, 4096, PROT_READ) = 0
mprotect(0x7f3a86f02000, 4096, PROT_READ) = 0
mprotect(0x7f3a866ab000, 4096, PROT_READ) = 0
mprotect(0x7f3a88bac000, 4096, PROT_READ) = 0
mprotect(0x7f3a89792000, 4096, PROT_READ) = 0
mprotect(0x7f3a86d98000, 16384, PROT_READ) = 0
fstat(0, {st_mode=S_IFCHR|0620, st_rdev=makedev(0x88, 0xa), ...}) = 0
fstat(1, {st_mode=S_IFCHR|0620, st_rdev=makedev(0x88, 0xa), ...}) = 0
fstat(2, {st_mode=S_IFREG|0644, st_size=135327, ...}) = 0
munmap(0x7f3a8858c000, 243639)          = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/datetime.py", {st_mode=S_IFREG|0644, st_size=88086, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/datetime.py", {st_mode=S_IFREG|0644, st_size=88086, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/datetime.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=56787, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36bd60)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=56787, ...}) = 0
brk(0x2b62a000)                         = 0x2b62a000
read(3, "o\r\r\n\0\0\0\0\20\272\367h\26X\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 56788) = 56787
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/math.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=259408, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/math.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=259408, ...}) = 0
mmap(NULL, 71728, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a89760000
mmap(0x7f3a89764000, 32768, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a89764000
mmap(0x7f3a8976c000, 16384, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xc000) = 0x7f3a8976c000
mmap(0x7f3a89770000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x10000) = 0x7f3a89770000
close(3)                                = 0
mprotect(0x7f3a89770000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_datetime.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=581056, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_datetime.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=581056, ...}) = 0
mmap(NULL, 126928, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a88b1c000
mmap(0x7f3a88b21000, 69632, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x5000) = 0x7f3a88b21000
mmap(0x7f3a88b32000, 24576, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x16000) = 0x7f3a88b32000
mmap(0x7f3a88b38000, 12288, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1b000) = 0x7f3a88b38000
close(3)                                = 0
mprotect(0x7f3a88b38000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_exceptions.py", {st_mode=S_IFREG|0644, st_size=5379, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_exceptions.py", {st_mode=S_IFREG|0644, st_size=5379, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/_exceptions.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5791, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36bdb0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5791, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\3\25\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5792) = 5791
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/dtypes.py", {st_mode=S_IFREG|0644, st_size=2229, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/dtypes.py", {st_mode=S_IFREG|0644, st_size=2229, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__pycache__/dtypes.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2254, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36bd20)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2254, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\265\10\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2255) = 2254
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2b64b000)                         = 0x2b64b000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/umath.py", {st_mode=S_IFREG|0644, st_size=2040, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/umath.py", {st_mode=S_IFREG|0644, st_size=2040, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/umath.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1696, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1696, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\370\7\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1697) = 1696
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/numerictypes.py", {st_mode=S_IFREG|0644, st_size=18098, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/numerictypes.py", {st_mode=S_IFREG|0644, st_size=18098, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/numerictypes.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=16976, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=16976, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\262F\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 16977) = 16976
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/numbers.py", {st_mode=S_IFREG|0644, st_size=10348, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/numbers.py", {st_mode=S_IFREG|0644, st_size=10348, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/numbers.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=11861, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36e310)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=11861, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hl(\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 11862) = 11861
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_string_helpers.py", {st_mode=S_IFREG|0644, st_size=2852, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_string_helpers.py", {st_mode=S_IFREG|0644, st_size=2852, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/_string_helpers.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3000, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36e310)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3000, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e$\v\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3001) = 3000
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_type_aliases.py", {st_mode=S_IFREG|0644, st_size=7534, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_type_aliases.py", {st_mode=S_IFREG|0644, st_size=7534, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/_type_aliases.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5326, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36e310)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5326, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302en\35\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5327) = 5326
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/compat/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36d1a0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/compat/__init__.abi3.so", 0x7ffeca36d1a0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/compat/__init__.so", 0x7ffeca36d1a0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/compat/__init__.py", {st_mode=S_IFREG|0644, st_size=448, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/compat/__init__.py", {st_mode=S_IFREG|0644, st_size=448, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/compat/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=615, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36d510)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=615, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\300\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 616) = 615
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/compat", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/compat", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/compat", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/compat", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b6393d0 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b6393d0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/compat/py3k.py", {st_mode=S_IFREG|0644, st_size=3833, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/compat/py3k.py", {st_mode=S_IFREG|0644, st_size=3833, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/compat/__pycache__/py3k.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4705, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36c060)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4705, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\371\16\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4706) = 4705
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/pickle.py", {st_mode=S_IFREG|0644, st_size=64949, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/pickle.py", {st_mode=S_IFREG|0644, st_size=64949, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/pickle.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=47136, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36b260)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=47136, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\265\375\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 47137) = 47136
read(3, "", 1)                          = 0
close(3)                                = 0
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a86700000
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/struct.py", {st_mode=S_IFREG|0644, st_size=257, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/struct.py", {st_mode=S_IFREG|0644, st_size=257, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/struct.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=302, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36a460)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=302, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\1\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 303) = 302
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_struct.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=219248, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_struct.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=219248, ...}) = 0
mmap(NULL, 60464, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a892f1000
mmap(0x7f3a892f5000, 20480, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a892f5000
mmap(0x7f3a892fa000, 16384, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x9000) = 0x7f3a892fa000
mmap(0x7f3a892fe000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xc000) = 0x7f3a892fe000
close(3)                                = 0
mprotect(0x7f3a892fe000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/_compat_pickle.py", {st_mode=S_IFREG|0644, st_size=8749, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/_compat_pickle.py", {st_mode=S_IFREG|0644, st_size=8749, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/_compat_pickle.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6126, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36a460)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6126, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\17\272\367h-\"\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6127) = 6126
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_pickle.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=537400, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_pickle.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=537400, ...}) = 0
mmap(NULL, 131272, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a885a7000
mmap(0x7f3a885ac000, 77824, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x5000) = 0x7f3a885ac000
mmap(0x7f3a885bf000, 24576, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x18000) = 0x7f3a885bf000
mmap(0x7f3a885c5000, 12288, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1e000) = 0x7f3a885c5000
close(3)                                = 0
mprotect(0x7f3a885c5000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_dtype.py", {st_mode=S_IFREG|0644, st_size=10606, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_dtype.py", {st_mode=S_IFREG|0644, st_size=10606, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/_dtype.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=8173, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36d510)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=8173, ...}) = 0
brk(0x2b66d000)                         = 0x2b66d000
read(3, "o\r\r\n\0\0\0\0.3\302en)\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 8174) = 8173
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/numeric.py", {st_mode=S_IFREG|0644, st_size=77014, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/numeric.py", {st_mode=S_IFREG|0644, st_size=77014, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/numeric.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=72966, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=72966, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\326,\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 72967) = 72966
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2b68e000)                         = 0x2b68e000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/shape_base.py", {st_mode=S_IFREG|0644, st_size=29743, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/shape_base.py", {st_mode=S_IFREG|0644, st_size=29743, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/shape_base.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=26567, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36dc60)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=26567, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e/t\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 26568) = 26567
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/fromnumeric.py", {st_mode=S_IFREG|0644, st_size=128821, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/fromnumeric.py", {st_mode=S_IFREG|0644, st_size=128821, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/fromnumeric.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=125651, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36c7b0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=125651, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e5\367\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 125652) = 125651
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2b6af000)                         = 0x2b6af000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_methods.py", {st_mode=S_IFREG|0644, st_size=8613, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_methods.py", {st_mode=S_IFREG|0644, st_size=8613, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/_methods.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5701, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36b300)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5701, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\245!\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5702) = 5701
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_ufunc_config.py", {st_mode=S_IFREG|0644, st_size=13944, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_ufunc_config.py", {st_mode=S_IFREG|0644, st_size=13944, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/_ufunc_config.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=14191, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36a500)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=14191, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302ex6\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 14192) = 14191
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/contextvars.py", {st_mode=S_IFREG|0644, st_size=129, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/contextvars.py", {st_mode=S_IFREG|0644, st_size=129, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/contextvars.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=500, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca369700)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=500, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\201\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 501) = 500
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_contextvars.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=27128, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_contextvars.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=27128, ...}) = 0
mmap(NULL, 16632, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a8961a000
mmap(0x7f3a8961b000, 4096, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1000) = 0x7f3a8961b000
mmap(0x7f3a8961c000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a8961c000
mmap(0x7f3a8961d000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a8961d000
close(3)                                = 0
mprotect(0x7f3a8961d000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/arrayprint.py", {st_mode=S_IFREG|0644, st_size=63608, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/arrayprint.py", {st_mode=S_IFREG|0644, st_size=63608, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/arrayprint.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=52343, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36dc60)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=52343, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302ex\370\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 52344) = 52343
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_asarray.py", {st_mode=S_IFREG|0644, st_size=3884, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_asarray.py", {st_mode=S_IFREG|0644, st_size=3884, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/_asarray.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3730, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36dc60)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3730, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e,\17\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3731) = 3730
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/defchararray.py", {st_mode=S_IFREG|0644, st_size=73617, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/defchararray.py", {st_mode=S_IFREG|0644, st_size=73617, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/defchararray.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=70962, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=70962, ...}) = 0
brk(0x2b6d5000)                         = 0x2b6d5000
read(3, "o\r\r\n\0\0\0\0.3\302e\221\37\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 70963) = 70962
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/records.py", {st_mode=S_IFREG|0644, st_size=37533, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/records.py", {st_mode=S_IFREG|0644, st_size=37533, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/records.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=30052, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=30052, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\235\222\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 30053) = 30052
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/memmap.py", {st_mode=S_IFREG|0644, st_size=11771, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/memmap.py", {st_mode=S_IFREG|0644, st_size=11771, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/memmap.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=10479, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f7c0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=10479, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\373-\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 10480) = 10479
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/function_base.py", {st_mode=S_IFREG|0644, st_size=19836, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/function_base.py", {st_mode=S_IFREG|0644, st_size=19836, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/function_base.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=17661, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=17661, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e|M\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 17662) = 17661
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_machar.py", {st_mode=S_IFREG|0644, st_size=11565, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_machar.py", {st_mode=S_IFREG|0644, st_size=11565, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/_machar.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=8287, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=8287, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e--\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 8288) = 8287
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/getlimits.py", {st_mode=S_IFREG|0644, st_size=25865, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/getlimits.py", {st_mode=S_IFREG|0644, st_size=25865, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/getlimits.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=19086, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=19086, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\te\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 19087) = 19086
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/einsumfunc.py", {st_mode=S_IFREG|0644, st_size=51868, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/einsumfunc.py", {st_mode=S_IFREG|0644, st_size=51868, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/einsumfunc.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=39589, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=39589, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\234\312\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 39590) = 39589
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_add_newdocs.py", {st_mode=S_IFREG|0644, st_size=208972, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_add_newdocs.py", {st_mode=S_IFREG|0644, st_size=208972, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/_add_newdocs.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=192336, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=192336, ...}) = 0
mmap(NULL, 192512, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a88578000
read(3, "o\r\r\n\0\0\0\0.3\302eL0\3\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 192337) = 192336
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2b6f6000)                         = 0x2b6f6000
munmap(0x7f3a88578000, 192512)          = 0
brk(0x2b717000)                         = 0x2b717000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_multiarray_tests.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=165096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_multiarray_tests.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=165096, ...}) = 0
mmap(NULL, 142544, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a88584000
mmap(0x7f3a8858d000, 77824, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x9000) = 0x7f3a8858d000
mmap(0x7f3a885a0000, 20480, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1c000) = 0x7f3a885a0000
mmap(0x7f3a885a5000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x21000) = 0x7f3a885a5000
close(3)                                = 0
mprotect(0x7f3a885a5000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_add_newdocs_scalars.py", {st_mode=S_IFREG|0644, st_size=12106, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_add_newdocs_scalars.py", {st_mode=S_IFREG|0644, st_size=12106, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/_add_newdocs_scalars.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=10973, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=10973, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302eJ/\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 10974) = 10973
read(3, "", 1)                          = 0
close(3)                                = 0
uname({sysname="Linux", nodename="loptop", ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_dtype_ctypes.py", {st_mode=S_IFREG|0644, st_size=3673, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_dtype_ctypes.py", {st_mode=S_IFREG|0644, st_size=3673, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/_dtype_ctypes.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3027, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3027, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302eY\16\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3028) = 3027
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_internal.py", {st_mode=S_IFREG|0644, st_size=28348, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/_internal.py", {st_mode=S_IFREG|0644, st_size=28348, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/core/__pycache__/_internal.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=23147, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=23147, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\274n\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 23148) = 23147
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ast.py", {st_mode=S_IFREG|0644, st_size=59900, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ast.py", {st_mode=S_IFREG|0644, st_size=59900, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/ast.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=55734, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36e310)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=55734, ...}) = 0
brk(0x2b740000)                         = 0x2b740000
read(3, "o\r\r\n\0\0\0\0\17\272\367h\374\351\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 55735) = 55734
read(3, "", 1)                          = 0
close(3)                                = 0
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a86300000
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ctypes/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36dfa0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ctypes/__init__.abi3.so", 0x7ffeca36dfa0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ctypes/__init__.so", 0x7ffeca36dfa0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ctypes/__init__.py", {st_mode=S_IFREG|0644, st_size=17988, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ctypes/__init__.py", {st_mode=S_IFREG|0644, st_size=17988, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/ctypes/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=15877, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36e310)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=15877, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hDF\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 15878) = 15877
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_ctypes.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=549032, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_ctypes.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=549032, ...}) = 0
mmap(NULL, 128744, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a88564000
mmap(0x7f3a8856b000, 57344, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x7000) = 0x7f3a8856b000
mmap(0x7f3a88579000, 24576, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x15000) = 0x7f3a88579000
mmap(0x7f3a8857f000, 20480, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1a000) = 0x7f3a8857f000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/../../glibc-hwcaps/x86-64-v3/libffi.so.8", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/../../glibc-hwcaps/x86-64-v3/", 0x7ffeca36c350, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/../../glibc-hwcaps/x86-64-v2/libffi.so.8", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/../../glibc-hwcaps/x86-64-v2/", 0x7ffeca36c350, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/../../libffi.so.8", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=72144, ...}) = 0
mmap(NULL, 67008, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a86df4000
mmap(0x7f3a86df7000, 40960, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a86df7000
mmap(0x7f3a86e01000, 8192, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xd000) = 0x7f3a86e01000
mmap(0x7f3a86e03000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xe000) = 0x7f3a86e03000
close(3)                                = 0
mprotect(0x7f3a86e03000, 4096, PROT_READ) = 0
mprotect(0x7f3a8857f000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ctypes", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ctypes", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ctypes", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/ctypes", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b727b20 /* 9 entries */, 32768) = 272
getdents64(3, 0x2b727b20 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ctypes/_endian.py", {st_mode=S_IFREG|0644, st_size=2000, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ctypes/_endian.py", {st_mode=S_IFREG|0644, st_size=2000, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/ctypes/__pycache__/_endian.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1896, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36d510)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1896, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\320\7\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1897) = 1896
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_pytesttester.py", {st_mode=S_IFREG|0644, st_size=6731, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_pytesttester.py", {st_mode=S_IFREG|0644, st_size=6731, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__pycache__/_pytesttester.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5835, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f7c0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5835, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302eK\32\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5836) = 5835
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__init__.abi3.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__init__.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__init__.py", {st_mode=S_IFREG|0644, st_size=2713, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__init__.py", {st_mode=S_IFREG|0644, st_size=2713, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2265, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2265, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\231\n\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2266) = 2265
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b735bb0 /* 49 entries */, 32768) = 1760
getdents64(3, 0x2b735bb0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/mixins.py", {st_mode=S_IFREG|0644, st_size=7071, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/mixins.py", {st_mode=S_IFREG|0644, st_size=7071, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/mixins.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=7009, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=7009, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\237\33\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 7010) = 7009
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/scimath.py", {st_mode=S_IFREG|0644, st_size=15037, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/scimath.py", {st_mode=S_IFREG|0644, st_size=15037, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/scimath.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=15619, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=15619, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\275:\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 15620) = 15619
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/type_check.py", {st_mode=S_IFREG|0644, st_size=19954, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/type_check.py", {st_mode=S_IFREG|0644, st_size=19954, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/type_check.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=19546, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f2d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=19546, ...}) = 0
brk(0x2b761000)                         = 0x2b761000
read(3, "o\r\r\n\0\0\0\0.3\302e\362M\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 19547) = 19546
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/ufunclike.py", {st_mode=S_IFREG|0644, st_size=6325, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/ufunclike.py", {st_mode=S_IFREG|0644, st_size=6325, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/ufunclike.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6238, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36e4d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6238, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\265\30\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6239) = 6238
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/index_tricks.py", {st_mode=S_IFREG|0644, st_size=31346, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/index_tricks.py", {st_mode=S_IFREG|0644, st_size=31346, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/index_tricks.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=29162, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=29162, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302erz\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 29163) = 29162
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/matrixlib/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36ef60) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/matrixlib/__init__.abi3.so", 0x7ffeca36ef60) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/matrixlib/__init__.so", 0x7ffeca36ef60) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/matrixlib/__init__.py", {st_mode=S_IFREG|0644, st_size=242, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/matrixlib/__init__.py", {st_mode=S_IFREG|0644, st_size=242, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/matrixlib/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=387, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f2d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=387, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\362\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 388) = 387
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/matrixlib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/matrixlib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/matrixlib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/matrixlib", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b749630 /* 9 entries */, 32768) = 280
getdents64(3, 0x2b749630 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/matrixlib/defmatrix.py", {st_mode=S_IFREG|0644, st_size=30656, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/matrixlib/defmatrix.py", {st_mode=S_IFREG|0644, st_size=30656, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/matrixlib/__pycache__/defmatrix.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=29582, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36de20)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=29582, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\300w\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 29583) = 29582
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36ccb0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/__init__.abi3.so", 0x7ffeca36ccb0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/__init__.so", 0x7ffeca36ccb0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/__init__.py", {st_mode=S_IFREG|0644, st_size=1813, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/__init__.py", {st_mode=S_IFREG|0644, st_size=1813, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1946, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36d020)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1946, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\25\7\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1947) = 1946
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b749630 /* 10 entries */, 32768) = 376
getdents64(3, 0x2b749630 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/linalg.py", {st_mode=S_IFREG|0644, st_size=90923, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/linalg.py", {st_mode=S_IFREG|0644, st_size=90923, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/__pycache__/linalg.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=83571, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36bb70)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=83571, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e+c\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 83572) = 83571
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2b782000)                         = 0x2b782000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/twodim_base.py", {st_mode=S_IFREG|0644, st_size=32947, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/twodim_base.py", {st_mode=S_IFREG|0644, st_size=32947, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/twodim_base.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=33078, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36ad70)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=33078, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\263\200\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 33079) = 33078
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/stride_tricks.py", {st_mode=S_IFREG|0644, st_size=17911, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/stride_tricks.py", {st_mode=S_IFREG|0644, st_size=17911, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/stride_tricks.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=16838, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca369f70)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=16838, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\367E\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 16839) = 16838
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/_umath_linalg.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=209680, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/_umath_linalg.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=209680, ...}) = 0
mmap(NULL, 186928, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a866d2000
mmap(0x7f3a866d9000, 131072, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x7000) = 0x7f3a866d9000
mmap(0x7f3a866f9000, 20480, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x27000) = 0x7f3a866f9000
mmap(0x7f3a866fe000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2c000) = 0x7f3a866fe000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/../../../../glibc-hwcaps/x86-64-v3/liblapack.so.3", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/../../../../glibc-hwcaps/x86-64-v3/", 0x7ffeca3694e0, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/../../../../glibc-hwcaps/x86-64-v2/liblapack.so.3", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/../../../../glibc-hwcaps/x86-64-v2/", 0x7ffeca3694e0, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/linalg/../../../../liblapack.so.3", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0@ \2\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=7650864, ...}) = 0
mmap(NULL, 7516360, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a85a00000
mprotect(0x7f3a85a22000, 7360512, PROT_NONE) = 0
mmap(0x7f3a85a22000, 6676480, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x22000) = 0x7f3a85a22000
mmap(0x7f3a86080000, 679936, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x680000) = 0x7f3a86080000
mmap(0x7f3a86127000, 20480, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x726000) = 0x7f3a86127000
close(3)                                = 0
mprotect(0x7f3a86127000, 16384, PROT_READ) = 0
mprotect(0x7f3a866fe000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36aa00) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/__init__.abi3.so", 0x7ffeca36aa00) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/__init__.so", 0x7ffeca36aa00) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/__init__.py", {st_mode=S_IFREG|0644, st_size=7003, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/__init__.py", {st_mode=S_IFREG|0644, st_size=7003, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6094, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36ad70)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6094, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e[\33\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6095) = 6094
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/__future__.py", {st_mode=S_IFREG|0644, st_size=5155, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/__future__.py", {st_mode=S_IFREG|0644, st_size=5155, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/__future__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4385, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca369f70)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4385, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\17\272\367h#\24\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4386) = 4385
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b76ca10 /* 16 entries */, 32768) = 560
getdents64(3, 0x2b76ca10 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_nested_sequence.py", {st_mode=S_IFREG|0644, st_size=2566, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_nested_sequence.py", {st_mode=S_IFREG|0644, st_size=2566, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/__pycache__/_nested_sequence.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3243, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca369f70)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3243, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\6\n\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3244) = 3243
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_nbit.py", {st_mode=S_IFREG|0644, st_size=345, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_nbit.py", {st_mode=S_IFREG|0644, st_size=345, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/__pycache__/_nbit.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=446, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca369f70)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=446, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302eY\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 447) = 446
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_char_codes.py", {st_mode=S_IFREG|0644, st_size=5916, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_char_codes.py", {st_mode=S_IFREG|0644, st_size=5916, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/__pycache__/_char_codes.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5071, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca369f70)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5071, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\34\27\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5072) = 5071
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_scalars.py", {st_mode=S_IFREG|0644, st_size=980, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_scalars.py", {st_mode=S_IFREG|0644, st_size=980, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/__pycache__/_scalars.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=736, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca369f70)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=736, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\324\3\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 737) = 736
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_shape.py", {st_mode=S_IFREG|0644, st_size=211, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_shape.py", {st_mode=S_IFREG|0644, st_size=211, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/__pycache__/_shape.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=318, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca369f70)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=318, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\323\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 319) = 318
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_dtype_like.py", {st_mode=S_IFREG|0644, st_size=5661, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_dtype_like.py", {st_mode=S_IFREG|0644, st_size=5661, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/__pycache__/_dtype_like.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3658, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca369f70)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3658, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\35\26\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3659) = 3658
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_array_like.py", {st_mode=S_IFREG|0644, st_size=4298, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/_array_like.py", {st_mode=S_IFREG|0644, st_size=4298, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/_typing/__pycache__/_array_like.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3313, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca369f70)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3313, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\312\20\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3314) = 3313
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/function_base.py", {st_mode=S_IFREG|0644, st_size=189172, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/function_base.py", {st_mode=S_IFREG|0644, st_size=189172, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/function_base.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=164758, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f2d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=164758, ...}) = 0
brk(0x2b7c5000)                         = 0x2b7c5000
read(3, "o\r\r\n\0\0\0\0.3\302e\364\342\2\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 164759) = 164758
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/histograms.py", {st_mode=S_IFREG|0644, st_size=37778, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/histograms.py", {st_mode=S_IFREG|0644, st_size=37778, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/histograms.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=30693, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36e4d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=30693, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\222\223\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 30694) = 30693
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/nanfunctions.py", {st_mode=S_IFREG|0644, st_size=65775, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/nanfunctions.py", {st_mode=S_IFREG|0644, st_size=65775, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/nanfunctions.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=58930, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=58930, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\357\0\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 58931) = 58930
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/shape_base.py", {st_mode=S_IFREG|0644, st_size=38947, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/shape_base.py", {st_mode=S_IFREG|0644, st_size=38947, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/shape_base.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=35649, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=35649, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e#\230\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 35650) = 35649
read(3, "", 1)                          = 0
close(3)                                = 0
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a86200000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/polynomial.py", {st_mode=S_IFREG|0644, st_size=44133, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/polynomial.py", {st_mode=S_IFREG|0644, st_size=44133, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/polynomial.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=41501, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=41501, ...}) = 0
brk(0x2b7ed000)                         = 0x2b7ed000
read(3, "o\r\r\n\0\0\0\0.3\302ee\254\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 41502) = 41501
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/utils.py", {st_mode=S_IFREG|0644, st_size=37804, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/utils.py", {st_mode=S_IFREG|0644, st_size=37804, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/utils.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=29214, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=29214, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\254\223\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 29215) = 29214
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/textwrap.py", {st_mode=S_IFREG|0644, st_size=19772, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/textwrap.py", {st_mode=S_IFREG|0644, st_size=19772, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/textwrap.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=14066, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f2d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=14066, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h<M\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 14067) = 14066
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/platform.py", {st_mode=S_IFREG|0755, st_size=42187, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/platform.py", {st_mode=S_IFREG|0755, st_size=42187, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/platform.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=27490, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f2d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=27490, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\313\244\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 27491) = 27490
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/subprocess.py", {st_mode=S_IFREG|0644, st_size=84917, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/subprocess.py", {st_mode=S_IFREG|0644, st_size=84917, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/subprocess.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=44736, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36e4d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=44736, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\265K\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 44737) = 44736
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/signal.py", {st_mode=S_IFREG|0644, st_size=2438, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/signal.py", {st_mode=S_IFREG|0644, st_size=2438, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/signal.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2930, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36d6d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2930, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\206\t\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2931) = 2930
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/threading.py", {st_mode=S_IFREG|0644, st_size=57200, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/threading.py", {st_mode=S_IFREG|0644, st_size=57200, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/threading.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=44964, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36d6d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=44964, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hp\337\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 44965) = 44964
read(3, "", 1)                          = 0
close(3)                                = 0
gettid()                                = 1266982
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/fcntl.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=57560, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/fcntl.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=57560, ...}) = 0
mmap(NULL, 29016, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a89612000
mmap(0x7f3a89614000, 8192, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a89614000
mmap(0x7f3a89616000, 8192, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a89616000
mmap(0x7f3a89618000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x5000) = 0x7f3a89618000
close(3)                                = 0
mprotect(0x7f3a89618000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_posixsubprocess.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=76368, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_posixsubprocess.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=76368, ...}) = 0
mmap(NULL, 24816, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a8855d000
mmap(0x7f3a8855f000, 8192, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a8855f000
mmap(0x7f3a88561000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a88561000
mmap(0x7f3a88562000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a88562000
close(3)                                = 0
mprotect(0x7f3a88562000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/select.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=113688, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/select.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=113688, ...}) = 0
mmap(NULL, 38304, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a86dea000
mmap(0x7f3a86ded000, 12288, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a86ded000
mmap(0x7f3a86df0000, 8192, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x6000) = 0x7f3a86df0000
mmap(0x7f3a86df2000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x8000) = 0x7f3a86df2000
close(3)                                = 0
mprotect(0x7f3a86df2000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/selectors.py", {st_mode=S_IFREG|0644, st_size=19536, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/selectors.py", {st_mode=S_IFREG|0644, st_size=19536, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/selectors.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=17359, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36d6d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=17359, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hPL\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 17360) = 17359
read(3, "", 1)                          = 0
close(3)                                = 0
mmap(NULL, 593920, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a8616f000
munmap(0x7f3a88bae000, 299008)          = 0
epoll_create1(EPOLL_CLOEXEC)            = 3
close(3)                                = 0
brk(0x2b816000)                         = 0x2b816000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/arraysetops.py", {st_mode=S_IFREG|0644, st_size=33655, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/arraysetops.py", {st_mode=S_IFREG|0644, st_size=33655, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/arraysetops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=27964, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=27964, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302ew\203\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 27965) = 27964
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/npyio.py", {st_mode=S_IFREG|0644, st_size=97316, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/npyio.py", {st_mode=S_IFREG|0644, st_size=97316, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/npyio.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=74626, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=74626, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e$|\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 74627) = 74626
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/format.py", {st_mode=S_IFREG|0644, st_size=34769, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/format.py", {st_mode=S_IFREG|0644, st_size=34769, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/format.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=26927, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36ec20)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=26927, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\321\207\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 26928) = 26927
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/_datasource.py", {st_mode=S_IFREG|0644, st_size=22631, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/_datasource.py", {st_mode=S_IFREG|0644, st_size=22631, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/_datasource.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=20411, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f2d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=20411, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302egX\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 20412) = 20411
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/_iotools.py", {st_mode=S_IFREG|0644, st_size=30868, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/_iotools.py", {st_mode=S_IFREG|0644, st_size=30868, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/_iotools.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=25877, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f2d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=25877, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\224x\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 25878) = 25877
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/arrayterator.py", {st_mode=S_IFREG|0644, st_size=7063, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/arrayterator.py", {st_mode=S_IFREG|0644, st_size=7063, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/arrayterator.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6996, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6996, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\227\33\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6997) = 6996
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/arraypad.py", {st_mode=S_IFREG|0644, st_size=31803, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/arraypad.py", {st_mode=S_IFREG|0644, st_size=31803, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/arraypad.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=22316, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=22316, ...}) = 0
brk(0x2b83b000)                         = 0x2b83b000
read(3, "o\r\r\n\0\0\0\0.3\302e;|\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 22317) = 22316
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/_version.py", {st_mode=S_IFREG|0644, st_size=4855, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/_version.py", {st_mode=S_IFREG|0644, st_size=4855, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/lib/__pycache__/_version.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4802, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4802, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\367\22\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4803) = 4802
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/__init__.abi3.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/__init__.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/__init__.py", {st_mode=S_IFREG|0644, st_size=8175, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/__init__.py", {st_mode=S_IFREG|0644, st_size=8175, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=8257, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=8257, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\357\37\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 8258) = 8257
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b81ffc0 /* 11 entries */, 32768) = 392
getdents64(3, 0x2b81ffc0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/_pocketfft.py", {st_mode=S_IFREG|0644, st_size=52897, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/_pocketfft.py", {st_mode=S_IFREG|0644, st_size=52897, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/__pycache__/_pocketfft.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=51785, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=51785, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\241\316\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 51786) = 51785
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/_pocketfft_internal.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=79808, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/_pocketfft_internal.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=79808, ...}) = 0
mmap(NULL, 78064, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a88be3000
mmap(0x7f3a88be4000, 61440, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1000) = 0x7f3a88be4000
mmap(0x7f3a88bf3000, 8192, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x10000) = 0x7f3a88bf3000
mmap(0x7f3a88bf5000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x11000) = 0x7f3a88bf5000
close(3)                                = 0
mprotect(0x7f3a88bf5000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/helper.py", {st_mode=S_IFREG|0644, st_size=6154, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/helper.py", {st_mode=S_IFREG|0644, st_size=6154, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/fft/__pycache__/helper.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6659, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6659, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\n\30\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6660) = 6659
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__init__.abi3.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__init__.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__init__.py", {st_mode=S_IFREG|0644, st_size=6781, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__init__.py", {st_mode=S_IFREG|0644, st_size=6781, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6841, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6841, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e}\32\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6842) = 6841
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
brk(0x2b85c000)                         = 0x2b85c000
getdents64(3, 0x2b833f40 /* 23 entries */, 32768) = 768
getdents64(3, 0x2b833f40 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/polynomial.py", {st_mode=S_IFREG|0644, st_size=49112, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/polynomial.py", {st_mode=S_IFREG|0644, st_size=49112, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__pycache__/polynomial.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=48662, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=48662, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\330\277\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 48663) = 48662
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/polyutils.py", {st_mode=S_IFREG|0644, st_size=23237, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/polyutils.py", {st_mode=S_IFREG|0644, st_size=23237, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__pycache__/polyutils.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=22668, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f2d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=22668, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\305Z\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 22669) = 22668
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/_polybase.py", {st_mode=S_IFREG|0644, st_size=39271, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/_polybase.py", {st_mode=S_IFREG|0644, st_size=39271, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__pycache__/_polybase.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=36395, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f980)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=36395, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302eg\231\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 36396) = 36395
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/chebyshev.py", {st_mode=S_IFREG|0644, st_size=62796, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/chebyshev.py", {st_mode=S_IFREG|0644, st_size=62796, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__pycache__/chebyshev.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=62225, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=62225, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302eL\365\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 62226) = 62225
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2b87d000)                         = 0x2b87d000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/legendre.py", {st_mode=S_IFREG|0644, st_size=51550, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/legendre.py", {st_mode=S_IFREG|0644, st_size=51550, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__pycache__/legendre.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=50896, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=50896, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e^\311\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 50897) = 50896
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/hermite.py", {st_mode=S_IFREG|0644, st_size=52514, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/hermite.py", {st_mode=S_IFREG|0644, st_size=52514, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__pycache__/hermite.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=52048, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=52048, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\"\315\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 52049) = 52048
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/hermite_e.py", {st_mode=S_IFREG|0644, st_size=52642, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/hermite_e.py", {st_mode=S_IFREG|0644, st_size=52642, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__pycache__/hermite_e.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=52033, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=52033, ...}) = 0
brk(0x2b8a3000)                         = 0x2b8a3000
read(3, "o\r\r\n\0\0\0\0.3\302e\242\315\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 52034) = 52033
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/laguerre.py", {st_mode=S_IFREG|0644, st_size=50858, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/laguerre.py", {st_mode=S_IFREG|0644, st_size=50858, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/polynomial/__pycache__/laguerre.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=50364, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=50364, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\252\306\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 50365) = 50364
read(3, "", 1)                          = 0
close(3)                                = 0
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a85900000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/__init__.abi3.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/__init__.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/__init__.py", {st_mode=S_IFREG|0644, st_size=7506, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/__init__.py", {st_mode=S_IFREG|0644, st_size=7506, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=7399, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=7399, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302eR\35\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 7400) = 7399
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b88c800 /* 31 entries */, 32768) = 1320
getdents64(3, 0x2b88c800 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_pickle.py", {st_mode=S_IFREG|0644, st_size=2318, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_pickle.py", {st_mode=S_IFREG|0644, st_size=2318, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/__pycache__/_pickle.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2198, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2198, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e\16\t\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2199) = 2198
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/mtrand.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=714496, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/mtrand.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=714496, ...}) = 0
mmap(NULL, 665032, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a8585d000
mmap(0x7f3a85862000, 303104, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x5000) = 0x7f3a85862000
mmap(0x7f3a858ac000, 327680, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4f000) = 0x7f3a858ac000
mmap(0x7f3a858fc000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x9e000) = 0x7f3a858fc000
mmap(0x7f3a858fe000, 5576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a858fe000
close(3)                                = 0
mprotect(0x7f3a858fc000, 4096, PROT_READ) = 0
brk(0x2b8c4000)                         = 0x2b8c4000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/bit_generator.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=224648, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/bit_generator.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=224648, ...}) = 0
mmap(NULL, 196248, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a88bb3000
mmap(0x7f3a88bb8000, 122880, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x5000) = 0x7f3a88bb8000
mmap(0x7f3a88bd6000, 40960, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x23000) = 0x7f3a88bd6000
mmap(0x7f3a88be0000, 12288, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2c000) = 0x7f3a88be0000
close(3)                                = 0
mprotect(0x7f3a88be0000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_common.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=246208, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_common.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=246208, ...}) = 0
mmap(NULL, 218272, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a86139000
mmap(0x7f3a8613d000, 176128, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a8613d000
mmap(0x7f3a86168000, 16384, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2f000) = 0x7f3a86168000
mmap(0x7f3a8616c000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x33000) = 0x7f3a8616c000
mmap(0x7f3a8616e000, 1184, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a8616e000
close(3)                                = 0
mprotect(0x7f3a8616c000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/secrets.py", {st_mode=S_IFREG|0644, st_size=2036, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/secrets.py", {st_mode=S_IFREG|0644, st_size=2036, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/secrets.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2170, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36d630)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2170, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\364\7\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2171) = 2170
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/base64.py", {st_mode=S_IFREG|0755, st_size=20847, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/base64.py", {st_mode=S_IFREG|0755, st_size=20847, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/base64.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=17157, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36c830)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=17157, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\17\272\367hoQ\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 17158) = 17157
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/binascii.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=122888, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/binascii.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=122888, ...}) = 0
mmap(NULL, 42096, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a86ddf000
mmap(0x7f3a86de1000, 20480, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a86de1000
mmap(0x7f3a86de6000, 8192, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x7000) = 0x7f3a86de6000
mmap(0x7f3a86de8000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x9000) = 0x7f3a86de8000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/../../libz.so.1", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=108688, ...}) = 0
mmap(NULL, 102424, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a866b8000
mmap(0x7f3a866bb000, 57344, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a866bb000
mmap(0x7f3a866c9000, 28672, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x11000) = 0x7f3a866c9000
mmap(0x7f3a866d0000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x17000) = 0x7f3a866d0000
close(3)                                = 0
mprotect(0x7f3a866d0000, 4096, PROT_READ) = 0
mprotect(0x7f3a86de8000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/hmac.py", {st_mode=S_IFREG|0644, st_size=7717, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/hmac.py", {st_mode=S_IFREG|0644, st_size=7717, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/hmac.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6968, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36c830)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6968, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h%\36\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6969) = 6968
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_hashlib.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=211256, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_hashlib.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=211256, ...}) = 0
mmap(NULL, 64848, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a8584d000
mmap(0x7f3a85851000, 24576, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a85851000
mmap(0x7f3a85857000, 16384, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xa000) = 0x7f3a85857000
mmap(0x7f3a8585b000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xe000) = 0x7f3a8585b000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/../../libcrypto.so.3", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=7194288, ...}) = 0
mmap(NULL, 6351048, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a85200000
mmap(0x7f3a852ee000, 3551232, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xee000) = 0x7f3a852ee000
mmap(0x7f3a85651000, 1269760, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x451000) = 0x7f3a85651000
mmap(0x7f3a85787000, 544768, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x586000) = 0x7f3a85787000
mmap(0x7f3a8580c000, 10440, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a8580c000
close(3)                                = 0
mprotect(0x7f3a85787000, 532480, PROT_READ) = 0
mprotect(0x7f3a8585b000, 4096, PROT_READ) = 0
futex(0x7f3a8580bd84, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580bd4c, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580b84c, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580bd40, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580bd38, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580be60, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580bd20, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580bd18, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580be08, FUTEX_WAKE_PRIVATE, 2147483647) = 0
brk(0x2b8e5000)                         = 0x2b8e5000
futex(0x7f3a8580bd8c, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580b798, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580bd30, FUTEX_WAKE_PRIVATE, 2147483647) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/ssl/openssl.cnf", O_RDONLY) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=12411, ...}) = 0
read(3, "#\n# OpenSSL example configuratio"..., 4096) = 4096
read(3, "d attributes must be the same, a"..., 4096) = 4096
read(3, "coding of an extension: beware e"..., 4096) = 4096
read(3, "cert = $insta::certout # insta.c"..., 4096) = 123
read(3, "", 4096)                       = 0
close(3)                                = 0
futex(0x7f3a8580b558, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580b59c, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580bcec, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580bce4, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580bcdc, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580b538, FUTEX_WAKE_PRIVATE, 2147483647) = 0
futex(0x7f3a8580bd10, FUTEX_WAKE_PRIVATE, 2147483647) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/hashlib.py", {st_mode=S_IFREG|0644, st_size=10229, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/hashlib.py", {st_mode=S_IFREG|0644, st_size=10229, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/hashlib.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=7099, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36ba30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=7099, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\365'\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 7100) = 7099
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_blake2.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=251824, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_blake2.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=251824, ...}) = 0
mmap(NULL, 54528, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a8583f000
mmap(0x7f3a85841000, 32768, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a85841000
mmap(0x7f3a85849000, 8192, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xa000) = 0x7f3a85849000
mmap(0x7f3a8584b000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xb000) = 0x7f3a8584b000
close(3)                                = 0
mprotect(0x7f3a8584b000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/random.py", {st_mode=S_IFREG|0644, st_size=33221, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/random.py", {st_mode=S_IFREG|0644, st_size=33221, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/random.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=22743, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36c830)       = -1 ENOTTY (Inappropriate ioctl for device)
brk(0x2b906000)                         = 0x2b906000
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=22743, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\305\201\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 22744) = 22743
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/bisect.py", {st_mode=S_IFREG|0644, st_size=3135, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/bisect.py", {st_mode=S_IFREG|0644, st_size=3135, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/bisect.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2583, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36ba30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2583, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\17\272\367h?\f\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2584) = 2583
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_bisect.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=62600, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_bisect.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=62600, ...}) = 0
mmap(NULL, 25184, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a88556000
mmap(0x7f3a88558000, 8192, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a88558000
mmap(0x7f3a8855a000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a8855a000
mmap(0x7f3a8855b000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a8855b000
close(3)                                = 0
mprotect(0x7f3a8855b000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_random.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=67448, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_random.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=67448, ...}) = 0
mmap(NULL, 25088, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a866b1000
mmap(0x7f3a866b3000, 8192, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a866b3000
mmap(0x7f3a866b5000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a866b5000
mmap(0x7f3a866b6000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a866b6000
close(3)                                = 0
mprotect(0x7f3a866b6000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_sha512.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=98160, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_sha512.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=98160, ...}) = 0
mmap(NULL, 37936, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a8612f000
mmap(0x7f3a86131000, 20480, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a86131000
mmap(0x7f3a86136000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x7000) = 0x7f3a86136000
mmap(0x7f3a86137000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x7000) = 0x7f3a86137000
close(3)                                = 0
mprotect(0x7f3a86137000, 4096, PROT_READ) = 0
getrandom("\xbb\x99\x02\x01\x87\x75\xa1\x33\xb6\x88\xb0\xdb\x99\x2f\x94\x42\xed\x10\x23\x39\xb1\x6a\xef\xa3\xf2\x30\xf1\x26\x5a\x3b\xae\x8e"..., 2496, GRND_NONBLOCK) = 2496
getrandom("\xdd\xfd\x02\x67\x0f\x84\x5b\x79\x4e\x16\x49\xef\xeb\xf5\x09\xbf\x73\x78\xb6\x49\xcb\x5e\x48\x26\xa9\x80\x22\x2c\x0c\x48\xf2\x88"..., 2496, GRND_NONBLOCK) = 2496
getrandom("\x0a\xb2\x70\x6f\xb8\x38\xba\x4b\xef\x52\x01\xf0\x9e\xaa\x97\xfa\xb4\xca\x5f\x7f\x91\xec\xea\x97\x3a\xc8\xd8\xc8\xb7\x15\x03\xee"..., 2496, GRND_NONBLOCK) = 2496
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_bounded_integers.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=357160, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_bounded_integers.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=357160, ...}) = 0
mmap(NULL, 324640, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a851b0000
mmap(0x7f3a851b5000, 258048, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x5000) = 0x7f3a851b5000
mmap(0x7f3a851f4000, 36864, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x44000) = 0x7f3a851f4000
mmap(0x7f3a851fd000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4c000) = 0x7f3a851fd000
mmap(0x7f3a851ff000, 1056, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a851ff000
close(3)                                = 0
mprotect(0x7f3a851fd000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_mt19937.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=102504, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_mt19937.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=102504, ...}) = 0
mmap(NULL, 92200, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a85828000
mmap(0x7f3a8582b000, 53248, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a8582b000
mmap(0x7f3a85838000, 20480, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x10000) = 0x7f3a85838000
mmap(0x7f3a8583d000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x14000) = 0x7f3a8583d000
close(3)                                = 0
mprotect(0x7f3a8583d000, 4096, PROT_READ) = 0
getrandom("\x12\xd6\xde\xd8\xdd\x56\xe8\xfa\x78\x2e\x0e\x28\xa6\x64\x2d\x73", 16, 0) = 16
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_philox.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=98248, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_philox.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=98248, ...}) = 0
mmap(NULL, 84008, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a85813000
mmap(0x7f3a85816000, 49152, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a85816000
mmap(0x7f3a85822000, 16384, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xf000) = 0x7f3a85822000
mmap(0x7f3a85826000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x13000) = 0x7f3a85826000
close(3)                                = 0
mprotect(0x7f3a85826000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_pcg64.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=113072, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_pcg64.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=113072, ...}) = 0
mmap(NULL, 96976, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a85198000
mmap(0x7f3a8519b000, 57344, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a8519b000
mmap(0x7f3a851a9000, 20480, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x11000) = 0x7f3a851a9000
mmap(0x7f3a851ae000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x16000) = 0x7f3a851ae000
close(3)                                = 0
mprotect(0x7f3a851ae000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_sfc64.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=67400, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_sfc64.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=67400, ...}) = 0
mmap(NULL, 59144, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a85189000
mmap(0x7f3a8518c000, 28672, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a8518c000
mmap(0x7f3a85193000, 12288, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xa000) = 0x7f3a85193000
mmap(0x7f3a85196000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xc000) = 0x7f3a85196000
close(3)                                = 0
mprotect(0x7f3a85196000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_generator.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=896784, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/random/_generator.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=896784, ...}) = 0
mmap(NULL, 832328, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a850bd000
mmap(0x7f3a850c5000, 442368, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x8000) = 0x7f3a850c5000
mmap(0x7f3a85131000, 339968, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x74000) = 0x7f3a85131000
mmap(0x7f3a85184000, 12288, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xc7000) = 0x7f3a85184000
mmap(0x7f3a85187000, 4936, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a85187000
close(3)                                = 0
mprotect(0x7f3a85184000, 4096, PROT_READ) = 0
brk(0x2b927000)                         = 0x2b927000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ctypeslib.py", {st_mode=S_IFREG|0644, st_size=17247, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ctypeslib.py", {st_mode=S_IFREG|0644, st_size=17247, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/__pycache__/ctypeslib.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=14479, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=14479, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e_C\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 14480) = 14479
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma/__init__.abi3.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma/__init__.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma/__init__.py", {st_mode=S_IFREG|0644, st_size=1404, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma/__init__.py", {st_mode=S_IFREG|0644, st_size=1404, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1530, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1530, ...}) = 0
read(3, "o\r\r\n\0\0\0\0.3\302e|\5\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1531) = 1530
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
brk(0x2b949000)                         = 0x2b949000
getdents64(3, 0x2b920050 /* 18 entries */, 32768) = 576
getdents64(3, 0x2b920050 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma/core.py", {st_mode=S_IFREG|0644, st_size=278213, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma/core.py", {st_mode=S_IFREG|0644, st_size=278213, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma/__pycache__/core.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=223725, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=223725, ...}) = 0
brk(0x2b977000)                         = 0x2b977000
read(3, "o\r\r\n\0\0\0\0.3\302e\305>\4\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 223726) = 223725
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/inspect.py", {st_mode=S_IFREG|0644, st_size=124378, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/inspect.py", {st_mode=S_IFREG|0644, st_size=124378, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/inspect.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=85407, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f2d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=85407, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\332\345\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 85408) = 85407
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/dis.py", {st_mode=S_IFREG|0644, st_size=20020, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/dis.py", {st_mode=S_IFREG|0644, st_size=20020, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/dis.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=15910, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36e4d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=15910, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h4N\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 15911) = 15910
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/opcode.py", {st_mode=S_IFREG|0644, st_size=5902, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/opcode.py", {st_mode=S_IFREG|0644, st_size=5902, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/opcode.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5701, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36d6d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5701, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\16\27\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5702) = 5701
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_opcode.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=29840, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_opcode.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=29840, ...}) = 0
mmap(NULL, 16688, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a892ec000
mmap(0x7f3a892ed000, 4096, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1000) = 0x7f3a892ed000
mmap(0x7f3a892ee000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a892ee000
mmap(0x7f3a892ef000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a892ef000
close(3)                                = 0
mprotect(0x7f3a892ef000, 4096, PROT_READ) = 0
brk(0x2b998000)                         = 0x2b998000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma/extras.py", {st_mode=S_IFREG|0644, st_size=64383, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma/extras.py", {st_mode=S_IFREG|0644, st_size=64383, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/numpy/ma/__pycache__/extras.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=57157, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3700d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=57157, ...}) = 0
brk(0x2b9bc000)                         = 0x2b9bc000
read(3, "o\r\r\n\0\0\0\0.3\302e\177\373\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 57158) = 57157
read(3, "", 1)                          = 0
close(3)                                = 0
uname({sysname="Linux", nodename="loptop", ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers/wrappers.py", {st_mode=S_IFREG|0644, st_size=4709, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers/wrappers.py", {st_mode=S_IFREG|0644, st_size=4709, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/data_helpers/__pycache__/wrappers.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6063, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6063, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215ie\22\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6064) = 6063
read(3, "", 1)                          = 0
close(3)                                = 0
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a84fbd000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/__init__.py", {st_mode=S_IFREG|0644, st_size=1066, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/__init__.py", {st_mode=S_IFREG|0644, st_size=1066, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=988, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=988, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i*\4\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 989) = 988
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9a2430 /* 12 entries */, 32768) = 352
getdents64(3, 0x2b9a2430 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/frontend.py", {st_mode=S_IFREG|0644, st_size=1401, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/frontend.py", {st_mode=S_IFREG|0644, st_size=1401, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/__pycache__/frontend.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2036, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2036, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215iy\5\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2037) = 2036
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/helpers/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/helpers/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/helpers/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/helpers/__init__.py", {st_mode=S_IFREG|0644, st_size=159, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/helpers/__init__.py", {st_mode=S_IFREG|0644, st_size=159, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/helpers/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=259, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=259, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\237\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 260) = 259
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/helpers", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/helpers", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/helpers", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/helpers", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9a3010 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9a3010 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/helpers/packing.py", {st_mode=S_IFREG|0644, st_size=4047, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/helpers/packing.py", {st_mode=S_IFREG|0644, st_size=4047, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/helpers/__pycache__/packing.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3399, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3399, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\317\17\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3400) = 3399
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/experimental/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/experimental/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/experimental/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/experimental/__init__.py", {st_mode=S_IFREG|0644, st_size=496, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/experimental/__init__.py", {st_mode=S_IFREG|0644, st_size=496, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/experimental/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=529, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=529, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\360\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 530) = 529
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/preprocess/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/preprocess/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/preprocess/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/preprocess/__init__.py", {st_mode=S_IFREG|0644, st_size=974, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/preprocess/__init__.py", {st_mode=S_IFREG|0644, st_size=974, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/preprocess/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=821, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=821, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\316\3\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 822) = 821
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/__init__.py", {st_mode=S_IFREG|0644, st_size=692, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/__init__.py", {st_mode=S_IFREG|0644, st_size=692, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=666, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=666, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\264\2\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 667) = 666
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9a4a60 /* 15 entries */, 32768) = 464
getdents64(3, 0x2b9a4a60 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/_properties.py", {st_mode=S_IFREG|0644, st_size=2326, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/_properties.py", {st_mode=S_IFREG|0644, st_size=2326, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/__pycache__/_properties.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2783, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2783, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\26\t\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2784) = 2783
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/hint/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/hint/__init__.abi3.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/hint/__init__.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/hint/__init__.py", {st_mode=S_IFREG|0644, st_size=608, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/hint/__init__.py", {st_mode=S_IFREG|0644, st_size=608, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/hint/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=577, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3738d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=577, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i`\2\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 578) = 577
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_cpu/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_cpu/__init__.abi3.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_cpu/__init__.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_cpu/__init__.py", {st_mode=S_IFREG|0644, st_size=287, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_cpu/__init__.py", {st_mode=S_IFREG|0644, st_size=287, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_cpu/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=405, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3738d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=405, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\37\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 406) = 405
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_gpu/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_gpu/__init__.abi3.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_gpu/__init__.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_gpu/__init__.py", {st_mode=S_IFREG|0644, st_size=431, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_gpu/__init__.py", {st_mode=S_IFREG|0644, st_size=431, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_gpu/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=474, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3738d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=474, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\257\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 475) = 474
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_auto/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_auto/__init__.abi3.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_auto/__init__.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_auto/__init__.py", {st_mode=S_IFREG|0644, st_size=370, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_auto/__init__.py", {st_mode=S_IFREG|0644, st_size=370, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/intel_auto/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=446, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3738d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=446, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215ir\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 447) = 446
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/device/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/device/__init__.abi3.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/device/__init__.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/device/__init__.py", {st_mode=S_IFREG|0644, st_size=416, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/device/__init__.py", {st_mode=S_IFREG|0644, st_size=416, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/device/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=453, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3738d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=453, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\240\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 454) = 453
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/log/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/log/__init__.abi3.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/log/__init__.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/log/__init__.py", {st_mode=S_IFREG|0644, st_size=332, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/log/__init__.py", {st_mode=S_IFREG|0644, st_size=332, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/log/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=409, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3738d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=409, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215iL\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 410) = 409
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/streams/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/streams/__init__.abi3.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/streams/__init__.so", 0x7ffeca373560) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/streams/__init__.py", {st_mode=S_IFREG|0644, st_size=348, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/streams/__init__.py", {st_mode=S_IFREG|0644, st_size=348, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/properties/streams/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=423, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3738d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=423, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\\\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 424) = 423
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op/__init__.py", {st_mode=S_IFREG|0644, st_size=595, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op/__init__.py", {st_mode=S_IFREG|0644, st_size=595, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=578, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=578, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215iS\2\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 579) = 578
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset1/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset1/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset1/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset1/__init__.py", {st_mode=S_IFREG|0644, st_size=4633, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset1/__init__.py", {st_mode=S_IFREG|0644, st_size=4633, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset1/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3618, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3618, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\31\22\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3619) = 3618
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset1", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset1", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset1", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset1", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9a74d0 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9a74d0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset1/ops.py", {st_mode=S_IFREG|0644, st_size=110562, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset1/ops.py", {st_mode=S_IFREG|0644, st_size=110562, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset1/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=98471, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=98471, ...}) = 0
brk(0x2b9e1000)                         = 0x2b9e1000
read(3, "o\r\r\n\0\0\0\0\220\263\215i\342\257\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 98472) = 98471
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/node_factory.py", {st_mode=S_IFREG|0644, st_size=5199, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/node_factory.py", {st_mode=S_IFREG|0644, st_size=5199, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/__pycache__/node_factory.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5276, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373180)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5276, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215iO\24\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5277) = 5276
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/exceptions.py", {st_mode=S_IFREG|0644, st_size=402, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/exceptions.py", {st_mode=S_IFREG|0644, st_size=402, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__pycache__/exceptions.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=764, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372380)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=764, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\222\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 765) = 764
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/decorators.py", {st_mode=S_IFREG|0644, st_size=6341, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/decorators.py", {st_mode=S_IFREG|0644, st_size=6341, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/__pycache__/decorators.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6056, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373180)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6056, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\305\30\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6057) = 6056
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/types.py", {st_mode=S_IFREG|0644, st_size=5305, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/types.py", {st_mode=S_IFREG|0644, st_size=5305, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/__pycache__/types.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4967, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372380)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4967, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\271\24\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4968) = 4967
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/logging/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/logging/__init__.abi3.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/logging/__init__.so", 0x7ffeca371210) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/logging/__init__.py", {st_mode=S_IFREG|0644, st_size=80232, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/logging/__init__.py", {st_mode=S_IFREG|0644, st_size=80232, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/logging/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=67145, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371580)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=67145, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367hh9\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 67146) = 67145
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/string.py", {st_mode=S_IFREG|0644, st_size=10566, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/string.py", {st_mode=S_IFREG|0644, st_size=10566, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/string.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=7097, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=7097, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hF)\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 7098) = 7097
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2ba10000)                         = 0x2ba10000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/input_validation.py", {st_mode=S_IFREG|0644, st_size=4803, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/input_validation.py", {st_mode=S_IFREG|0644, st_size=4803, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/utils/__pycache__/input_validation.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4366, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373180)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4366, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\303\22\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4367) = 4366
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset2/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset2/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset2/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset2/__init__.py", {st_mode=S_IFREG|0644, st_size=4887, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset2/__init__.py", {st_mode=S_IFREG|0644, st_size=4887, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset2/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3833, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3833, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\27\23\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3834) = 3833
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset2", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset2", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset2", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset2", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9ce1d0 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9ce1d0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset2/ops.py", {st_mode=S_IFREG|0644, st_size=7634, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset2/ops.py", {st_mode=S_IFREG|0644, st_size=7634, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset2/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=7039, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=7039, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\322\35\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 7040) = 7039
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset3/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset3/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset3/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset3/__init__.py", {st_mode=S_IFREG|0644, st_size=5642, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset3/__init__.py", {st_mode=S_IFREG|0644, st_size=5642, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset3/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4431, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4431, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\n\26\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4432) = 4431
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset3", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset3", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset3", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset3", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9cf3f0 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9cf3f0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset3/ops.py", {st_mode=S_IFREG|0644, st_size=24245, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset3/ops.py", {st_mode=S_IFREG|0644, st_size=24245, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset3/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=21564, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=21564, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\265^\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 21565) = 21564
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset4/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset4/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset4/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset4/__init__.py", {st_mode=S_IFREG|0644, st_size=6086, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset4/__init__.py", {st_mode=S_IFREG|0644, st_size=6086, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset4/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4786, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4786, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\306\27\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4787) = 4786
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset4", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset4", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset4", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset4", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9d6f60 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9d6f60 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset4/ops.py", {st_mode=S_IFREG|0644, st_size=18101, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset4/ops.py", {st_mode=S_IFREG|0644, st_size=18101, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset4/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=16360, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=16360, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\265F\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 16361) = 16360
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset5/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset5/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset5/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset5/__init__.py", {st_mode=S_IFREG|0644, st_size=6424, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset5/__init__.py", {st_mode=S_IFREG|0644, st_size=6424, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset5/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5065, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5065, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\30\31\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5066) = 5065
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset5", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset5", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset5", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset5", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9d8180 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9d8180 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset5/ops.py", {st_mode=S_IFREG|0644, st_size=14826, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset5/ops.py", {st_mode=S_IFREG|0644, st_size=14826, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset5/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=12829, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=12829, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\3529\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 12830) = 12829
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset6/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset6/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset6/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset6/__init__.py", {st_mode=S_IFREG|0644, st_size=6531, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset6/__init__.py", {st_mode=S_IFREG|0644, st_size=6531, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset6/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5173, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5173, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\203\31\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5174) = 5173
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset6", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset6", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset6", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset6", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9dc9e0 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9dc9e0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset6/ops.py", {st_mode=S_IFREG|0644, st_size=7782, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset6/ops.py", {st_mode=S_IFREG|0644, st_size=7782, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset6/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6323, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6323, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215if\36\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6324) = 6323
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9dce40 /* 6 entries */, 32768) = 168
getdents64(3, 0x2b9dce40 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op/util/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca372e10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op/util/__init__.abi3.so", 0x7ffeca372e10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op/util/__init__.so", 0x7ffeca372e10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op/util/__init__.py", {st_mode=S_IFREG|0644, st_size=1000, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op/util/__init__.py", {st_mode=S_IFREG|0644, st_size=1000, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/op/util/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=873, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373180)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=873, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\350\3\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 874) = 873
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset7/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset7/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset7/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset7/__init__.py", {st_mode=S_IFREG|0644, st_size=6680, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset7/__init__.py", {st_mode=S_IFREG|0644, st_size=6680, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset7/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5303, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5303, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\30\32\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5304) = 5303
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset7", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset7", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset7", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset7", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9dfcc0 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9dfcc0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset7/ops.py", {st_mode=S_IFREG|0644, st_size=5014, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset7/ops.py", {st_mode=S_IFREG|0644, st_size=5014, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset7/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4752, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4752, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\226\23\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4753) = 4752
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset8/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset8/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset8/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset8/__init__.py", {st_mode=S_IFREG|0644, st_size=7169, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset8/__init__.py", {st_mode=S_IFREG|0644, st_size=7169, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset8/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5703, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5703, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\1\34\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5704) = 5703
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset8", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset8", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset8", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset8", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9e0ee0 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9e0ee0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset8/ops.py", {st_mode=S_IFREG|0644, st_size=33167, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset8/ops.py", {st_mode=S_IFREG|0644, st_size=33167, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset8/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=29633, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=29633, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\217\201\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 29634) = 29633
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset9/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset9/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset9/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset9/__init__.py", {st_mode=S_IFREG|0644, st_size=7416, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset9/__init__.py", {st_mode=S_IFREG|0644, st_size=7416, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset9/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5911, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5911, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\370\34\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5912) = 5911
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset9", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset9", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset9", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset9", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9ecfe0 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9ecfe0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset9/ops.py", {st_mode=S_IFREG|0644, st_size=13739, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset9/ops.py", {st_mode=S_IFREG|0644, st_size=13739, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset9/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=11991, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=11991, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\2535\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 11992) = 11991
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset10/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset10/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset10/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset10/__init__.py", {st_mode=S_IFREG|0644, st_size=7580, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset10/__init__.py", {st_mode=S_IFREG|0644, st_size=7580, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset10/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6053, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6053, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\234\35\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6054) = 6053
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset10", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9efa10 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9efa10 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset10/ops.py", {st_mode=S_IFREG|0644, st_size=7318, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset10/ops.py", {st_mode=S_IFREG|0644, st_size=7318, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset10/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6680, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6680, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\226\34\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6681) = 6680
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset11/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset11/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset11/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset11/__init__.py", {st_mode=S_IFREG|0644, st_size=7581, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset11/__init__.py", {st_mode=S_IFREG|0644, st_size=7581, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset11/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6075, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6075, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\235\35\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6076) = 6075
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset11", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset11", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset11", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset11", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9f3060 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9f3060 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset11/ops.py", {st_mode=S_IFREG|0644, st_size=4488, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset11/ops.py", {st_mode=S_IFREG|0644, st_size=4488, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset11/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4183, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4183, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\210\21\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4184) = 4183
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset12/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset12/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset12/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset12/__init__.py", {st_mode=S_IFREG|0644, st_size=7636, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset12/__init__.py", {st_mode=S_IFREG|0644, st_size=7636, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset12/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6139, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6139, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\324\35\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6140) = 6139
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset12", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset12", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset12", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset12", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9f66b0 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9f66b0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset12/ops.py", {st_mode=S_IFREG|0644, st_size=4417, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset12/ops.py", {st_mode=S_IFREG|0644, st_size=4417, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset12/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4174, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4174, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215iA\21\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4175) = 4174
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset13/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset13/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset13/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset13/__init__.py", {st_mode=S_IFREG|0644, st_size=8016, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset13/__init__.py", {st_mode=S_IFREG|0644, st_size=8016, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset13/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6450, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6450, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215iP\37\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6451) = 6450
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset13", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset13", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset13", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset13", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9f8ae0 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9f8ae0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset13/ops.py", {st_mode=S_IFREG|0644, st_size=16936, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset13/ops.py", {st_mode=S_IFREG|0644, st_size=16936, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset13/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=15079, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=15079, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i(B\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 15080) = 15079
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset14/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset14/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset14/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset14/__init__.py", {st_mode=S_IFREG|0644, st_size=8114, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset14/__init__.py", {st_mode=S_IFREG|0644, st_size=8114, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset14/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6546, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6546, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\262\37\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6547) = 6546
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset14", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset14", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset14", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset14", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2b9fff60 /* 7 entries */, 32768) = 208
getdents64(3, 0x2b9fff60 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset14/ops.py", {st_mode=S_IFREG|0644, st_size=7016, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset14/ops.py", {st_mode=S_IFREG|0644, st_size=7016, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset14/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6290, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6290, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215ih\33\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6291) = 6290
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset15/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset15/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset15/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset15/__init__.py", {st_mode=S_IFREG|0644, st_size=8727, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset15/__init__.py", {st_mode=S_IFREG|0644, st_size=8727, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset15/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6990, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6990, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\27\"\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6991) = 6990
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset15", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset15", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset15", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset15", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2ba02a60 /* 7 entries */, 32768) = 208
getdents64(3, 0x2ba02a60 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset15/ops.py", {st_mode=S_IFREG|0644, st_size=14665, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset15/ops.py", {st_mode=S_IFREG|0644, st_size=14665, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset15/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=13289, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=13289, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215iI9\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 13290) = 13289
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset16/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset16/__init__.abi3.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset16/__init__.so", 0x7ffeca374a10) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset16/__init__.py", {st_mode=S_IFREG|0644, st_size=528, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset16/__init__.py", {st_mode=S_IFREG|0644, st_size=528, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset16/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=400, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374d80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=400, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\20\2\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 401) = 400
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset16", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset16", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset16", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset16", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2ba06960 /* 7 entries */, 32768) = 208
getdents64(3, 0x2ba06960 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset16/ops.py", {st_mode=S_IFREG|0644, st_size=8313, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset16/ops.py", {st_mode=S_IFREG|0644, st_size=8313, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/opset16/__pycache__/ops.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=7624, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373f80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=7624, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215iy \0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 7625) = 7624
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374850) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/__init__.abi3.so", 0x7ffeca374850) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/__init__.so", 0x7ffeca374850) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/__init__.py", {st_mode=S_IFREG|0644, st_size=148, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/__init__.py", {st_mode=S_IFREG|0644, st_size=148, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=244, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374bc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=244, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\224\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 245) = 244
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca374460) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/__init__.abi3.so", 0x7ffeca374460) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/__init__.so", 0x7ffeca374460) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/__init__.py", {st_mode=S_IFREG|0644, st_size=148, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/openvino.tools.pkg", 0x7ffeca3749a0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2ba06960 /* 7 entries */, 32768) = 200
getdents64(3, 0x2ba06960 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca3750c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__init__.abi3.so", 0x7ffeca3750c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__init__.so", 0x7ffeca3750c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__init__.py", {st_mode=S_IFREG|0644, st_size=1443, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__init__.py", {st_mode=S_IFREG|0644, st_size=1443, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1147, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca375430)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1147, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\243\5\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1148) = 1147
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2ba06960 /* 34 entries */, 32768) = 1200
getdents64(3, 0x2ba06960 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/convert.py", {st_mode=S_IFREG|0644, st_size=4799, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/convert.py", {st_mode=S_IFREG|0644, st_size=4799, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__pycache__/convert.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4759, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374630)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4759, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\277\22\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4760) = 4759
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/cli_parser.py", {st_mode=S_IFREG|0644, st_size=25011, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/cli_parser.py", {st_mode=S_IFREG|0644, st_size=25011, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__pycache__/cli_parser.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=17752, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373830)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=17752, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\263a\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 17753) = 17752
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/argparse.py", {st_mode=S_IFREG|0644, st_size=98543, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/argparse.py", {st_mode=S_IFREG|0644, st_size=98543, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/argparse.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=63480, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=63480, ...}) = 0
brk(0x2ba38000)                         = 0x2ba38000
read(3, "o\r\r\n\0\0\0\0\17\272\367h\357\200\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 63481) = 63480
read(3, "", 1)                          = 0
close(3)                                = 0
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a84ebd000
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/gettext.py", {st_mode=S_IFREG|0644, st_size=27266, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/gettext.py", {st_mode=S_IFREG|0644, st_size=27266, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/gettext.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=18105, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=18105, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\202j\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 18106) = 18105
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/error.py", {st_mode=S_IFREG|0644, st_size=1463, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/error.py", {st_mode=S_IFREG|0644, st_size=1463, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__pycache__/error.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1770, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1770, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\267\5\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1771) = 1770
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/help.py", {st_mode=S_IFREG|0644, st_size=2791, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/help.py", {st_mode=S_IFREG|0644, st_size=2791, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__pycache__/help.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2184, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2184, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\347\n\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2185) = 2184
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca371e50) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__init__.abi3.so", 0x7ffeca371e50) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__init__.so", 0x7ffeca371e50) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__init__.py", {st_mode=S_IFREG|0644, st_size=82, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__init__.py", {st_mode=S_IFREG|0644, st_size=82, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=194, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3721c0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=194, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215iR\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 195) = 194
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2ba184f0 /* 30 entries */, 32768) = 1160
getdents64(3, 0x2ba184f0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/shape_utils.py", {st_mode=S_IFREG|0644, st_size=4025, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/shape_utils.py", {st_mode=S_IFREG|0644, st_size=4025, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__pycache__/shape_utils.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2608, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2608, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\271\17\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2609) = 2608
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/type_utils.py", {st_mode=S_IFREG|0644, st_size=2753, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/type_utils.py", {st_mode=S_IFREG|0644, st_size=2753, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__pycache__/type_utils.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1761, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1761, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\301\n\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1762) = 1761
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/utils.py", {st_mode=S_IFREG|0644, st_size=7647, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/utils.py", {st_mode=S_IFREG|0644, st_size=7647, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__pycache__/utils.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=7250, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=7250, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\337\35\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 7251) = 7250
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca3718c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/__init__.abi3.so", 0x7ffeca3718c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/__init__.so", 0x7ffeca3718c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/__init__.py", {st_mode=S_IFREG|0644, st_size=181, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/__init__.py", {st_mode=S_IFREG|0644, st_size=181, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=279, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=279, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\265\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 280) = 279
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2ba184f0 /* 8 entries */, 32768) = 248
getdents64(3, 0x2ba184f0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/main.py", {st_mode=S_IFREG|0644, st_size=15447, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/main.py", {st_mode=S_IFREG|0644, st_size=15447, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/__pycache__/main.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=10605, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370e30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=10605, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215iW<\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 10606) = 10605
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36f450) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/__init__.abi3.so", 0x7ffeca36f450) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/__init__.so", 0x7ffeca36f450) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/__init__.py", {st_mode=S_IFREG|0644, st_size=136, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/__init__.py", {st_mode=S_IFREG|0644, st_size=136, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=239, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f7c0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=239, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\210\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 240) = 239
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2ba184f0 /* 7 entries */, 32768) = 224
getdents64(3, 0x2ba184f0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/backend_ga.py", {st_mode=S_IFREG|0644, st_size=3984, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/backend_ga.py", {st_mode=S_IFREG|0644, st_size=3984, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/__pycache__/backend_ga.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4182, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36e9c0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4182, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\220\17\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4183) = 4182
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/uuid.py", {st_mode=S_IFREG|0644, st_size=27500, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/uuid.py", {st_mode=S_IFREG|0644, st_size=27500, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/uuid.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=22752, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36dbc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=22752, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hlk\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 22753) = 22752
read(3, "", 1)                          = 0
close(3)                                = 0
uname({sysname="Linux", nodename="loptop", ...}) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_uuid.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=26752, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_uuid.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=26752, ...}) = 0
mmap(NULL, 16632, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a88bae000
mmap(0x7f3a88baf000, 4096, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1000) = 0x7f3a88baf000
mmap(0x7f3a88bb0000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a88bb0000
mmap(0x7f3a88bb1000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a88bb1000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/../../libuuid.so.1", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=35944, ...}) = 0
mmap(NULL, 32808, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84eb4000
mmap(0x7f3a84eb6000, 16384, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a84eb6000
mmap(0x7f3a84eba000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x6000) = 0x7f3a84eba000
mmap(0x7f3a84ebb000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x6000) = 0x7f3a84ebb000
close(3)                                = 0
mprotect(0x7f3a84ebb000, 4096, PROT_READ) = 0
mprotect(0x7f3a88bb1000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib/request.py", {st_mode=S_IFREG|0644, st_size=101742, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib/request.py", {st_mode=S_IFREG|0644, st_size=101742, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/urllib/__pycache__/request.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=71578, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36d510)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=71578, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367hn\215\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 71579) = 71578
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36c3a0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/__init__.abi3.so", 0x7ffeca36c3a0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/__init__.so", 0x7ffeca36c3a0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/__init__.py", {st_mode=S_IFREG|0644, st_size=1766, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/__init__.py", {st_mode=S_IFREG|0644, st_size=1766, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1555, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36c710)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1555, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\346\6\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1556) = 1555
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/http/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36bb30) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/http/__init__.abi3.so", 0x7ffeca36bb30) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/http/__init__.so", 0x7ffeca36bb30) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/http/__init__.py", {st_mode=S_IFREG|0644, st_size=6733, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/http/__init__.py", {st_mode=S_IFREG|0644, st_size=6733, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/http/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6411, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36bea0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6411, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367hM\32\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6412) = 6411
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/http", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/http", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/http", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/http", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2ba23220 /* 8 entries */, 32768) = 240
getdents64(3, 0x2ba23220 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/http/client.py", {st_mode=S_IFREG|0644, st_size=56795, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/http/client.py", {st_mode=S_IFREG|0644, st_size=56795, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/http/__pycache__/client.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=35432, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36c710)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=35432, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h\333\335\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 35433) = 35432
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2ba23220 /* 25 entries */, 32768) = 864
getdents64(3, 0x2ba23220 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/parser.py", {st_mode=S_IFREG|0644, st_size=5041, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/parser.py", {st_mode=S_IFREG|0644, st_size=5041, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/parser.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5927, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36b910)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5927, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\261\23\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5928) = 5927
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/feedparser.py", {st_mode=S_IFREG|0644, st_size=22780, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/feedparser.py", {st_mode=S_IFREG|0644, st_size=22780, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/feedparser.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=10573, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36ab10)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=10573, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\374X\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 10574) = 10573
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/errors.py", {st_mode=S_IFREG|0644, st_size=3814, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/errors.py", {st_mode=S_IFREG|0644, st_size=3814, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/errors.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6104, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca369660)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6104, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\346\16\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6105) = 6104
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/_policybase.py", {st_mode=S_IFREG|0644, st_size=15534, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/_policybase.py", {st_mode=S_IFREG|0644, st_size=15534, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/_policybase.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=15494, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca369d10)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=15494, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\256<\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 15495) = 15494
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/header.py", {st_mode=S_IFREG|0644, st_size=24102, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/header.py", {st_mode=S_IFREG|0644, st_size=24102, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/header.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=16716, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca368860)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=16716, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h&^\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 16717) = 16716
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/quoprimime.py", {st_mode=S_IFREG|0644, st_size=9858, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/quoprimime.py", {st_mode=S_IFREG|0644, st_size=9858, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/quoprimime.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=7877, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca367a60)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=7877, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\202&\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 7878) = 7877
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/base64mime.py", {st_mode=S_IFREG|0644, st_size=3559, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/base64mime.py", {st_mode=S_IFREG|0644, st_size=3559, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/base64mime.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3235, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca367a60)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3235, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\347\r\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3236) = 3235
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/charset.py", {st_mode=S_IFREG|0644, st_size=17128, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/charset.py", {st_mode=S_IFREG|0644, st_size=17128, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/charset.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=11570, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3673b0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=11570, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\350B\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 11571) = 11570
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/encoders.py", {st_mode=S_IFREG|0644, st_size=1786, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/encoders.py", {st_mode=S_IFREG|0644, st_size=1786, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/encoders.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1881, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3665b0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1881, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\372\6\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1882) = 1881
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/quopri.py", {st_mode=S_IFREG|0755, st_size=7268, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/quopri.py", {st_mode=S_IFREG|0755, st_size=7268, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/quopri.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5789, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3657b0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5789, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hd\34\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5790) = 5789
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2ba59000)                         = 0x2ba59000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/utils.py", {st_mode=S_IFREG|0644, st_size=17201, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/utils.py", {st_mode=S_IFREG|0644, st_size=17201, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/utils.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=12120, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca368f10)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=12120, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h1C\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 12121) = 12120
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/socket.py", {st_mode=S_IFREG|0644, st_size=37006, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/socket.py", {st_mode=S_IFREG|0644, st_size=37006, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/socket.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=29030, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca368110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=29030, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\216\220\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 29031) = 29030
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_socket.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=284016, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_socket.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=284016, ...}) = 0
mmap(NULL, 97176, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84e9c000
mmap(0x7f3a84ea1000, 36864, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x5000) = 0x7f3a84ea1000
mmap(0x7f3a84eaa000, 32768, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xe000) = 0x7f3a84eaa000
mmap(0x7f3a84eb2000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x16000) = 0x7f3a84eb2000
close(3)                                = 0
mprotect(0x7f3a84eb2000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/array.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=232128, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/array.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=232128, ...}) = 0
mmap(NULL, 67592, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84e8b000
mmap(0x7f3a84e8f000, 28672, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a84e8f000
mmap(0x7f3a84e96000, 16384, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xb000) = 0x7f3a84e96000
mmap(0x7f3a84e9a000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xf000) = 0x7f3a84e9a000
close(3)                                = 0
mprotect(0x7f3a84e9a000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/_parseaddr.py", {st_mode=S_IFREG|0644, st_size=17821, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/_parseaddr.py", {st_mode=S_IFREG|0644, st_size=17821, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/_parseaddr.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=12507, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca368110)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=12507, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\235E\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 12508) = 12507
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/calendar.py", {st_mode=S_IFREG|0644, st_size=24575, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/calendar.py", {st_mode=S_IFREG|0644, st_size=24575, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/calendar.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=26557, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca367310)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=26557, ...}) = 0
brk(0x2ba7a000)                         = 0x2ba7a000
read(3, "o\r\r\n\0\0\0\0\17\272\367h\377_\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 26558) = 26557
read(3, "", 1)                          = 0
close(3)                                = 0
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a84d8b000
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/locale.py", {st_mode=S_IFREG|0644, st_size=78124, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/locale.py", {st_mode=S_IFREG|0644, st_size=78124, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/locale.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=46160, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca366510)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=46160, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h,1\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 46161) = 46160
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/message.py", {st_mode=S_IFREG|0644, st_size=47060, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/message.py", {st_mode=S_IFREG|0644, st_size=47060, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/message.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=37799, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36b910)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=37799, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\324\267\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 37800) = 37799
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/uu.py", {st_mode=S_IFREG|0644, st_size=7301, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/uu.py", {st_mode=S_IFREG|0644, st_size=7301, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/uu.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3862, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36ab10)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3862, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\205\34\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3863) = 3862
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/_encoded_words.py", {st_mode=S_IFREG|0644, st_size=8541, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/_encoded_words.py", {st_mode=S_IFREG|0644, st_size=8541, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/_encoded_words.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5994, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36ab10)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5994, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h]!\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5995) = 5994
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/iterators.py", {st_mode=S_IFREG|0644, st_size=2135, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/email/iterators.py", {st_mode=S_IFREG|0644, st_size=2135, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/email/__pycache__/iterators.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1963, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36a890)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1963, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hW\10\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1964) = 1963
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ssl.py", {st_mode=S_IFREG|0644, st_size=53895, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/ssl.py", {st_mode=S_IFREG|0644, st_size=53895, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/ssl.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=45276, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36b910)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=45276, ...}) = 0
brk(0x2ba9e000)                         = 0x2ba9e000
read(3, "o\r\r\n\0\0\0\0\20\272\367h\207\322\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 45277) = 45276
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_ssl.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=500328, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_ssl.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=500328, ...}) = 0
mmap(NULL, 212360, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84d57000
mmap(0x7f3a84d6a000, 40960, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x13000) = 0x7f3a84d6a000
mmap(0x7f3a84d74000, 57344, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1d000) = 0x7f3a84d74000
mmap(0x7f3a84d82000, 36864, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2b000) = 0x7f3a84d82000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/../../libssl.so.3", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=1202288, ...}) = 0
mmap(NULL, 1029960, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84c5b000
mmap(0x7f3a84c7f000, 630784, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x24000) = 0x7f3a84c7f000
mmap(0x7f3a84d19000, 196608, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xbe000) = 0x7f3a84d19000
mmap(0x7f3a84d49000, 57344, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xee000) = 0x7f3a84d49000
close(3)                                = 0
mprotect(0x7f3a84d49000, 40960, PROT_READ) = 0
mprotect(0x7f3a84d82000, 4096, PROT_READ) = 0
brk(0x2bac2000)                         = 0x2bac2000
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/tempfile.py", {st_mode=S_IFREG|0644, st_size=29469, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/tempfile.py", {st_mode=S_IFREG|0644, st_size=29469, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/tempfile.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=24567, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36c710)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=24567, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\35s\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 24568) = 24567
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/shutil.py", {st_mode=S_IFREG|0644, st_size=54572, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/shutil.py", {st_mode=S_IFREG|0644, st_size=54572, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/shutil.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=38790, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36b910)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=38790, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h,\325\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 38791) = 38790
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/zlib.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=138144, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/zlib.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=138144, ...}) = 0
mmap(NULL, 46864, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84c4f000
mmap(0x7f3a84c52000, 16384, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a84c52000
mmap(0x7f3a84c56000, 12288, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x7000) = 0x7f3a84c56000
mmap(0x7f3a84c59000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x9000) = 0x7f3a84c59000
close(3)                                = 0
mprotect(0x7f3a84c59000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/bz2.py", {st_mode=S_IFREG|0644, st_size=11847, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/bz2.py", {st_mode=S_IFREG|0644, st_size=11847, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/bz2.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=10865, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36ab10)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=10865, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\17\272\367hG.\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 10866) = 10865
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/_compression.py", {st_mode=S_IFREG|0644, st_size=5681, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/_compression.py", {st_mode=S_IFREG|0644, st_size=5681, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/_compression.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4507, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca369d10)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4507, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\17\272\367h1\26\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4508) = 4507
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_bz2.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=89016, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_bz2.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=89016, ...}) = 0
mmap(NULL, 29584, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84c47000
mmap(0x7f3a84c49000, 8192, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a84c49000
mmap(0x7f3a84c4b000, 8192, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a84c4b000
mmap(0x7f3a84c4d000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x5000) = 0x7f3a84c4d000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/../../libbz2.so.1.0", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=229016, ...}) = 0
mmap(NULL, 80976, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84c33000
mmap(0x7f3a84c35000, 57344, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a84c35000
mmap(0x7f3a84c43000, 8192, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x10000) = 0x7f3a84c43000
mmap(0x7f3a84c45000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x11000) = 0x7f3a84c45000
close(3)                                = 0
mprotect(0x7f3a84c45000, 4096, PROT_READ) = 0
mprotect(0x7f3a84c4d000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lzma.py", {st_mode=S_IFREG|0644, st_size=13277, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lzma.py", {st_mode=S_IFREG|0644, st_size=13277, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/lzma.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=12354, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36ab10)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=12354, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\3353\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 12355) = 12354
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_lzma.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=145496, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_lzma.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=145496, ...}) = 0
mmap(NULL, 46576, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84c27000
mmap(0x7f3a84c2a000, 16384, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a84c2a000
mmap(0x7f3a84c2e000, 12288, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x7000) = 0x7f3a84c2e000
mmap(0x7f3a84c31000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x9000) = 0x7f3a84c31000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/../../liblzma.so.5", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=222712, ...}) = 0
mmap(NULL, 196824, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84bf6000
mmap(0x7f3a84bfb000, 122880, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x5000) = 0x7f3a84bfb000
mmap(0x7f3a84c19000, 49152, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x23000) = 0x7f3a84c19000
mmap(0x7f3a84c25000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2f000) = 0x7f3a84c25000
close(3)                                = 0
mprotect(0x7f3a84c25000, 4096, PROT_READ) = 0
mprotect(0x7f3a84c31000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib/error.py", {st_mode=S_IFREG|0644, st_size=2415, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib/error.py", {st_mode=S_IFREG|0644, st_size=2415, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/urllib/__pycache__/error.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3107, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36c710)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3107, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367ho\t\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3108) = 3107
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib/response.py", {st_mode=S_IFREG|0644, st_size=2361, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/urllib/response.py", {st_mode=S_IFREG|0644, st_size=2361, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/urllib/__pycache__/response.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3462, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36b910)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3462, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h9\t\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3463) = 3462
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2bae3000)                         = 0x2bae3000
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a84af6000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/backend.py", {st_mode=S_IFREG|0644, st_size=3039, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/backend.py", {st_mode=S_IFREG|0644, st_size=3039, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/__pycache__/backend.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4601, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36dbc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4601, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\337\v\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4602) = 4601
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36c1e0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__init__.abi3.so", 0x7ffeca36c1e0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__init__.so", 0x7ffeca36c1e0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__init__.py", {st_mode=S_IFREG|0644, st_size=82, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__init__.py", {st_mode=S_IFREG|0644, st_size=82, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=187, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36c550)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=187, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215iR\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 188) = 187
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bac8520 /* 12 entries */, 32768) = 408
getdents64(3, 0x2bac8520 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/message.py", {st_mode=S_IFREG|0644, st_size=354, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/message.py", {st_mode=S_IFREG|0644, st_size=354, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__pycache__/message.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=816, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36cdc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=816, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215ib\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 817) = 816
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/cid.py", {st_mode=S_IFREG|0644, st_size=2553, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/cid.py", {st_mode=S_IFREG|0644, st_size=2553, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__pycache__/cid.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2645, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36dbc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2645, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\371\t\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2646) = 2645
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/opt_in_checker.py", {st_mode=S_IFREG|0644, st_size=12766, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/opt_in_checker.py", {st_mode=S_IFREG|0644, st_size=12766, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__pycache__/opt_in_checker.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=10015, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36cdc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=10015, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\3361\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 10016) = 10015
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/colored_print.py", {st_mode=S_IFREG|0644, st_size=1160, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/colored_print.py", {st_mode=S_IFREG|0644, st_size=1160, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__pycache__/colored_print.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1163, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36bfc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1163, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\210\4\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1164) = 1163
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/input_with_timeout.py", {st_mode=S_IFREG|0644, st_size=1959, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/input_with_timeout.py", {st_mode=S_IFREG|0644, st_size=1959, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__pycache__/input_with_timeout.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1715, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36bfc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1715, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\247\7\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1716) = 1715
read(3, "", 1)                          = 0
close(3)                                = 0
lstat("/home", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
lstat("/home/loganr", {st_mode=S_IFDIR|0700, st_size=4096, ...}) = 0
lstat("/home/loganr/.conda", {st_mode=S_IFDIR|S_ISGID|0775, st_size=4096, ...}) = 0
lstat("/home/loganr/.conda/envs", {st_mode=S_IFDIR|S_ISGID|0775, st_size=4096, ...}) = 0
lstat("/home/loganr/.conda/envs/fly", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
lstat("/home/loganr/.conda/envs/fly/lib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=98304, ...}) = 0
lstat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
lstat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
lstat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
lstat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
lstat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/opt_in_checker.py", {st_mode=S_IFREG|0644, st_size=12766, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/backend_ga4.py", {st_mode=S_IFREG|0644, st_size=5687, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/backend_ga4.py", {st_mode=S_IFREG|0644, st_size=5687, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/backend/__pycache__/backend_ga4.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5457, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36e9c0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5457, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i7\26\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5458) = 5457
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36d850) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json/__init__.abi3.so", 0x7ffeca36d850) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json/__init__.so", 0x7ffeca36d850) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json/__init__.py", {st_mode=S_IFREG|0644, st_size=14020, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json/__init__.py", {st_mode=S_IFREG|0644, st_size=14020, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/json/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=12259, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36dbc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=12259, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h\3046\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 12260) = 12259
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/json", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bad2c10 /* 8 entries */, 32768) = 240
getdents64(3, 0x2bad2c10 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json/decoder.py", {st_mode=S_IFREG|0644, st_size=12473, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json/decoder.py", {st_mode=S_IFREG|0644, st_size=12473, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/json/__pycache__/decoder.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=9761, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36cdc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=9761, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h\2710\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 9762) = 9761
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json/scanner.py", {st_mode=S_IFREG|0644, st_size=2425, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json/scanner.py", {st_mode=S_IFREG|0644, st_size=2425, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/json/__pycache__/scanner.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2170, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36b910)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2170, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367hy\t\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2171) = 2170
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_json.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=163184, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_json.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=163184, ...}) = 0
mmap(NULL, 50832, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84ae9000
mmap(0x7f3a84aec000, 24576, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a84aec000
mmap(0x7f3a84af2000, 8192, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x9000) = 0x7f3a84af2000
mmap(0x7f3a84af4000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xa000) = 0x7f3a84af4000
close(3)                                = 0
mprotect(0x7f3a84af4000, 4096, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json/encoder.py", {st_mode=S_IFREG|0644, st_size=16074, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/json/encoder.py", {st_mode=S_IFREG|0644, st_size=16074, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/json/__pycache__/encoder.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=11112, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36cdc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=11112, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h\312>\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 11113) = 11112
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/copy.py", {st_mode=S_IFREG|0644, st_size=8681, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/copy.py", {st_mode=S_IFREG|0644, st_size=8681, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/copy.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6991, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36dbc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6991, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\351!\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6992) = 6991
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/params.py", {st_mode=S_IFREG|0644, st_size=146, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/params.py", {st_mode=S_IFREG|0644, st_size=146, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__pycache__/params.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=243, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36dbc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=243, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\222\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 244) = 243
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/sender.py", {st_mode=S_IFREG|0644, st_size=2232, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/sender.py", {st_mode=S_IFREG|0644, st_size=2232, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__pycache__/sender.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2461, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370030)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2461, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\270\10\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2462) = 2461
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36eec0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/__init__.abi3.so", 0x7ffeca36eec0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/__init__.so", 0x7ffeca36eec0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/__init__.py", {st_mode=S_IFREG|0644, st_size=38, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/__init__.py", {st_mode=S_IFREG|0644, st_size=38, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=130, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f230)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=130, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h&\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 131) = 130
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/concurrent", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bada720 /* 5 entries */, 32768) = 144
getdents64(3, 0x2bada720 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36e810) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures/__init__.abi3.so", 0x7ffeca36e810) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures/__init__.so", 0x7ffeca36e810) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures/__init__.py", {st_mode=S_IFREG|0644, st_size=1554, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures/__init__.py", {st_mode=S_IFREG|0644, st_size=1554, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1361, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36eb80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1361, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\22\6\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1362) = 1361
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bada720 /* 7 entries */, 32768) = 208
getdents64(3, 0x2bada720 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures/_base.py", {st_mode=S_IFREG|0644, st_size=22848, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures/_base.py", {st_mode=S_IFREG|0644, st_size=22848, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures/__pycache__/_base.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=22228, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36dd80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=22228, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h@Y\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 22229) = 22228
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/stats_processor.py", {st_mode=S_IFREG|0644, st_size=3118, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/stats_processor.py", {st_mode=S_IFREG|0644, st_size=3118, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino_telemetry/utils/__pycache__/stats_processor.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3420, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370030)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3420, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i.\f\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3421) = 3420
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca3718c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/__init__.abi3.so", 0x7ffeca3718c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/__init__.so", 0x7ffeca3718c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/__init__.py", {st_mode=S_IFREG|0644, st_size=30596, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/__init__.py", {st_mode=S_IFREG|0644, st_size=30596, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=37091, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=37091, ...}) = 0
brk(0x2bb0c000)                         = 0x2bb0c000
read(3, "o\r\r\n\0\0\0\0\21\272\367h\204w\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 37092) = 37091
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/csv.py", {st_mode=S_IFREG|0644, st_size=16030, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/csv.py", {st_mode=S_IFREG|0644, st_size=16030, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/csv.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=11793, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370e30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=11793, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\236>\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 11794) = 11793
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_csv.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=142816, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_csv.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=142816, ...}) = 0
mmap(NULL, 51184, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84adc000
mmap(0x7f3a84adf000, 20480, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a84adf000
mmap(0x7f3a84ae4000, 12288, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x8000) = 0x7f3a84ae4000
mmap(0x7f3a84ae7000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xa000) = 0x7f3a84ae7000
close(3)                                = 0
mprotect(0x7f3a84ae7000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/zipfile.py", {st_mode=S_IFREG|0644, st_size=90860, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/zipfile.py", {st_mode=S_IFREG|0644, st_size=90860, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/zipfile.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=62109, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370e30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=62109, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\354b\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 62110) = 62109
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2baf2090 /* 10 entries */, 32768) = 328
getdents64(3, 0x2baf2090 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/_adapters.py", {st_mode=S_IFREG|0644, st_size=1862, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/_adapters.py", {st_mode=S_IFREG|0644, st_size=1862, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/__pycache__/_adapters.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2339, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2339, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367hF\7\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2340) = 2339
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/_text.py", {st_mode=S_IFREG|0644, st_size=2198, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/_text.py", {st_mode=S_IFREG|0644, st_size=2198, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/__pycache__/_text.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3314, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f980)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3314, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h\226\10\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3315) = 3314
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/_functools.py", {st_mode=S_IFREG|0644, st_size=2895, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/_functools.py", {st_mode=S_IFREG|0644, st_size=2895, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/__pycache__/_functools.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3104, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36eb80)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3104, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367hO\v\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3105) = 3104
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/_meta.py", {st_mode=S_IFREG|0644, st_size=1130, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/_meta.py", {st_mode=S_IFREG|0644, st_size=1130, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/__pycache__/_meta.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2518, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2518, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367hj\4\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2519) = 2518
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/_collections.py", {st_mode=S_IFREG|0644, st_size=743, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/_collections.py", {st_mode=S_IFREG|0644, st_size=743, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/__pycache__/_collections.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1773, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370e30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1773, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h\347\2\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1774) = 1773
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/_itertools.py", {st_mode=S_IFREG|0644, st_size=607, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/_itertools.py", {st_mode=S_IFREG|0644, st_size=607, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib/metadata/__pycache__/_itertools.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=566, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370e30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=566, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h_\2\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 567) = 566
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/abc.py", {st_mode=S_IFREG|0644, st_size=14421, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/importlib/abc.py", {st_mode=S_IFREG|0644, st_size=14421, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/importlib/__pycache__/abc.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=15886, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370e30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=15886, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367hU8\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 15887) = 15886
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/convert_impl.py", {st_mode=S_IFREG|0644, st_size=23965, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/convert_impl.py", {st_mode=S_IFREG|0644, st_size=23965, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__pycache__/convert_impl.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=15696, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373830)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=15696, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\235]\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 15697) = 15696
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/tracemalloc.py", {st_mode=S_IFREG|0644, st_size=18047, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/tracemalloc.py", {st_mode=S_IFREG|0644, st_size=18047, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/tracemalloc.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=17520, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=17520, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\177F\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 17521) = 17520
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/check_config.py", {st_mode=S_IFREG|0644, st_size=2040, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/check_config.py", {st_mode=S_IFREG|0644, st_size=2040, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__pycache__/check_config.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1676, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1676, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\370\7\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1677) = 1676
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/pipeline.py", {st_mode=S_IFREG|0644, st_size=14102, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/pipeline.py", {st_mode=S_IFREG|0644, st_size=14102, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__pycache__/pipeline.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=8832, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=8832, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\0267\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 8833) = 8832
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/analysis.py", {st_mode=S_IFREG|0644, st_size=2010, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/analysis.py", {st_mode=S_IFREG|0644, st_size=2010, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__pycache__/analysis.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2031, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2031, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\332\7\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2032) = 2031
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/extractor.py", {st_mode=S_IFREG|0644, st_size=18316, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/extractor.py", {st_mode=S_IFREG|0644, st_size=18316, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__pycache__/extractor.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=11015, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=11015, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\214G\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 11016) = 11015
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2bb2d000)                         = 0x2bb2d000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/moc_emit_ir.py", {st_mode=S_IFREG|0644, st_size=1424, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/moc_emit_ir.py", {st_mode=S_IFREG|0644, st_size=1424, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__pycache__/moc_emit_ir.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1059, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1059, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\220\5\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1060) = 1059
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/preprocessing.py", {st_mode=S_IFREG|0644, st_size=10253, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/preprocessing.py", {st_mode=S_IFREG|0644, st_size=10253, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__pycache__/preprocessing.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6957, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6957, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\r(\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6958) = 6957
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/layout_utils.py", {st_mode=S_IFREG|0644, st_size=3250, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/layout_utils.py", {st_mode=S_IFREG|0644, st_size=3250, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__pycache__/layout_utils.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2484, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370e30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2484, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\262\f\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2485) = 2484
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/get_ov_update_message.py", {st_mode=S_IFREG|0644, st_size=739, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/get_ov_update_message.py", {st_mode=S_IFREG|0644, st_size=739, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__pycache__/get_ov_update_message.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=849, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=849, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\343\2\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 850) = 849
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/version.py", {st_mode=S_IFREG|0644, st_size=2554, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/version.py", {st_mode=S_IFREG|0644, st_size=2554, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__pycache__/version.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2695, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2695, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\372\t\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2696) = 2695
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/logger.py", {st_mode=S_IFREG|0644, st_size=3172, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/logger.py", {st_mode=S_IFREG|0644, st_size=3172, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__pycache__/logger.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3054, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3054, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215id\f\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3055) = 3054
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca371cf0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/__init__.abi3.so", 0x7ffeca371cf0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/__init__.so", 0x7ffeca371cf0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/__init__.py", {st_mode=S_IFREG|0644, st_size=607, ...}) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca371050) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/__init__.abi3.so", 0x7ffeca371050) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/__init__.so", 0x7ffeca371050) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/__init__.py", {st_mode=S_IFREG|0644, st_size=607, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/__init__.py", {st_mode=S_IFREG|0644, st_size=607, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=188, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3713c0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=188, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\265o\370h_\2\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 189) = 188
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bb0c000 /* 11 entries */, 32768) = 344
getdents64(3, 0x2bb0c000 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/logging/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca3718c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/logging/__init__.abi3.so", 0x7ffeca3718c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/logging/__init__.so", 0x7ffeca3718c0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/logging/__init__.py", {st_mode=S_IFREG|0644, st_size=43583, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/logging/__init__.py", {st_mode=S_IFREG|0644, st_size=43583, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/logging/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=38833, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=38833, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\265o\370h?\252\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 38834) = 38833
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/getpass.py", {st_mode=S_IFREG|0644, st_size=5990, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/getpass.py", {st_mode=S_IFREG|0644, st_size=5990, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/getpass.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=4464, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370e30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=4464, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hf\27\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 4465) = 4464
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/termios.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=66800, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/termios.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=66800, ...}) = 0
mmap(NULL, 33008, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84ad3000
mmap(0x7f3a84ad6000, 4096, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a84ad6000
mmap(0x7f3a84ad7000, 8192, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a84ad7000
mmap(0x7f3a84ad9000, 12288, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x5000) = 0x7f3a84ad9000
close(3)                                = 0
mprotect(0x7f3a84ad9000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/timeit.py", {st_mode=S_IFREG|0755, st_size=13495, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/timeit.py", {st_mode=S_IFREG|0755, st_size=13495, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/timeit.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=12023, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370e30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=12023, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\2674\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 12024) = 12023
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca370410) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__init__.abi3.so", 0x7ffeca370410) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__init__.so", 0x7ffeca370410) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__init__.py", {st_mode=S_IFREG|0644, st_size=7665, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__init__.py", {st_mode=S_IFREG|0644, st_size=7665, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3651, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3651, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\265o\370h\361\35\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3652) = 3651
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bb18a70 /* 13 entries */, 32768) = 456
getdents64(3, 0x2bb18a70 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_argument_parser.py", {st_mode=S_IFREG|0644, st_size=20629, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_argument_parser.py", {st_mode=S_IFREG|0644, st_size=20629, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__pycache__/_argument_parser.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=22345, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f2d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=22345, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\265o\370h\225P\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 22346) = 22345
read(3, "", 1)                          = 0
close(3)                                = 0
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f3a849d3000
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36d8f0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/__init__.abi3.so", 0x7ffeca36d8f0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/__init__.so", 0x7ffeca36d8f0) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/__init__.py", {st_mode=S_IFREG|0644, st_size=557, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/__init__.py", {st_mode=S_IFREG|0644, st_size=557, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/xml/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=947, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36dc60)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=947, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h-\2\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 948) = 947
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/xml", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bb18a70 /* 8 entries */, 32768) = 224
getdents64(3, 0x2bb18a70 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca36e160) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/__init__.abi3.so", 0x7ffeca36e160) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/__init__.so", 0x7ffeca36e160) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/__init__.py", {st_mode=S_IFREG|0644, st_size=4019, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/__init__.py", {st_mode=S_IFREG|0644, st_size=4019, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5531, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36e4d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5531, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h\263\17\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5532) = 5531
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bb1d2e0 /* 11 entries */, 32768) = 368
getdents64(3, 0x2bb1d2e0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/domreg.py", {st_mode=S_IFREG|0644, st_size=3451, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/domreg.py", {st_mode=S_IFREG|0644, st_size=3451, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/__pycache__/domreg.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2854, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36d6d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2854, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h{\r\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2855) = 2854
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/minidom.py", {st_mode=S_IFREG|0644, st_size=68140, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/minidom.py", {st_mode=S_IFREG|0644, st_size=68140, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/__pycache__/minidom.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=55583, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36de20)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=55583, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h,\n\1\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 55584) = 55583
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2bb4e000)                         = 0x2bb4e000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/minicompat.py", {st_mode=S_IFREG|0644, st_size=3367, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/minicompat.py", {st_mode=S_IFREG|0644, st_size=3367, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/__pycache__/minicompat.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2944, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36d020)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2944, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h'\r\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2945) = 2944
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/xmlbuilder.py", {st_mode=S_IFREG|0644, st_size=12387, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/xmlbuilder.py", {st_mode=S_IFREG|0644, st_size=12387, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/__pycache__/xmlbuilder.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=12555, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36d020)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=12555, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367hc0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 12556) = 12555
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/NodeFilter.py", {st_mode=S_IFREG|0644, st_size=936, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/NodeFilter.py", {st_mode=S_IFREG|0644, st_size=936, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/xml/dom/__pycache__/NodeFilter.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=1222, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36c220)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=1222, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\21\272\367h\250\3\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 1223) = 1222
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_helpers.py", {st_mode=S_IFREG|0644, st_size=14154, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_helpers.py", {st_mode=S_IFREG|0644, st_size=14154, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__pycache__/_helpers.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=10058, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36de20)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=10058, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\265o\370hJ7\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 10059) = 10058
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2bb7d000)                         = 0x2bb7d000
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_defines.py", {st_mode=S_IFREG|0644, st_size=52895, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_defines.py", {st_mode=S_IFREG|0644, st_size=52895, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__pycache__/_defines.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=38907, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36f2d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=38907, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\265o\370h\237\316\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 38908) = 38907
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_exceptions.py", {st_mode=S_IFREG|0644, st_size=3619, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_exceptions.py", {st_mode=S_IFREG|0644, st_size=3619, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__pycache__/_exceptions.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3777, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36de20)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3777, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\265o\370h#\16\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3778) = 3777
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_flag.py", {st_mode=S_IFREG|0644, st_size=20079, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_flag.py", {st_mode=S_IFREG|0644, st_size=20079, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__pycache__/_flag.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=18524, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36de20)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=18524, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\265o\370hoN\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 18525) = 18524
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_flagvalues.py", {st_mode=S_IFREG|0644, st_size=54364, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_flagvalues.py", {st_mode=S_IFREG|0644, st_size=54364, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__pycache__/_flagvalues.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=43845, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36de20)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=43845, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\265o\370h\\\324\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 43846) = 43845
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_validators_classes.py", {st_mode=S_IFREG|0644, st_size=6093, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_validators_classes.py", {st_mode=S_IFREG|0644, st_size=6093, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__pycache__/_validators_classes.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6711, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36c970)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6711, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\265o\370h\315\27\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6712) = 6711
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_validators.py", {st_mode=S_IFREG|0644, st_size=14140, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/_validators.py", {st_mode=S_IFREG|0644, st_size=14140, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/flags/__pycache__/_validators.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=13319, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca36de20)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=13319, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\265o\370h<7\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 13320) = 13319
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/logging", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/logging", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/logging", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/logging", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bb4e050 /* 6 entries */, 32768) = 176
getdents64(3, 0x2bb4e050 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/logging/converter.py", {st_mode=S_IFREG|0644, st_size=6323, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/logging/converter.py", {st_mode=S_IFREG|0644, st_size=6323, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/absl/logging/__pycache__/converter.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5222, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370780)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5222, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\265o\370h\263\30\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5223) = 5222
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/telemetry_utils.py", {st_mode=S_IFREG|0644, st_size=4642, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/telemetry_utils.py", {st_mode=S_IFREG|0644, st_size=4642, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__pycache__/telemetry_utils.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=3583, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=3583, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\"\22\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 3584) = 3583
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/telemetry_params.py", {st_mode=S_IFREG|0644, st_size=132, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/telemetry_params.py", {st_mode=S_IFREG|0644, st_size=132, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__pycache__/telemetry_params.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=233, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=233, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\204\0\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 234) = 233
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/pytorch_frontend_utils.py", {st_mode=S_IFREG|0644, st_size=10194, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/pytorch_frontend_utils.py", {st_mode=S_IFREG|0644, st_size=10194, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__pycache__/pytorch_frontend_utils.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=6872, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=6872, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\322'\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 6873) = 6872
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/paddle_frontend_utils.py", {st_mode=S_IFREG|0644, st_size=3392, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/paddle_frontend_utils.py", {st_mode=S_IFREG|0644, st_size=3392, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__pycache__/paddle_frontend_utils.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2788, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2788, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i@\r\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2789) = 2788
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/jax_frontend_utils.py", {st_mode=S_IFREG|0644, st_size=489, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/jax_frontend_utils.py", {st_mode=S_IFREG|0644, st_size=489, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/moc_frontend/__pycache__/jax_frontend_utils.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=616, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=616, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\351\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 617) = 616
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca371e50) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/__init__.abi3.so", 0x7ffeca371e50) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/__init__.so", 0x7ffeca371e50) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/__init__.py", {st_mode=S_IFREG|0644, st_size=669, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/__init__.py", {st_mode=S_IFREG|0644, st_size=669, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=686, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3721c0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=686, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215i\235\2\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 687) = 686
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bb52f20 /* 11 entries */, 32768) = 416
getdents64(3, 0x2bb52f20 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/py_tensorflow_frontend.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0644, st_size=625281, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/py_tensorflow_frontend.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0 \240\1\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0644, st_size=625281, ...}) = 0
mmap(NULL, 629376, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84939000
mprotect(0x7f3a84952000, 495616, PROT_NONE) = 0
mmap(0x7f3a84952000, 344064, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x19000) = 0x7f3a84952000
mmap(0x7f3a849a6000, 122880, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x6d000) = 0x7f3a849a6000
mmap(0x7f3a849c5000, 24576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x8b000) = 0x7f3a849c5000
mmap(0x7f3a849cb000, 32768, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x91000) = 0x7f3a849cb000
close(3)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/../../../openvino/libs/glibc-hwcaps/x86-64-v3/libopenvino_tensorflow_frontend.so.2540", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/../../../openvino/libs/glibc-hwcaps/x86-64-v3/", 0x7ffeca370200, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/../../../openvino/libs/glibc-hwcaps/x86-64-v2/libopenvino_tensorflow_frontend.so.2540", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/../../../openvino/libs/glibc-hwcaps/x86-64-v2/", 0x7ffeca370200, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/../../../openvino/libs/libopenvino_tensorflow_frontend.so.2540", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0`*\6\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0644, st_size=5155617, ...}) = 0
mmap(NULL, 5184288, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84400000
mprotect(0x7f3a84460000, 4730880, PROT_NONE) = 0
mmap(0x7f3a84460000, 3719168, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x60000) = 0x7f3a84460000
mmap(0x7f3a847ec000, 868352, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3ec000) = 0x7f3a847ec000
mmap(0x7f3a848c1000, 110592, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4c0000) = 0x7f3a848c1000
mmap(0x7f3a848dc000, 28120, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a848dc000
mmap(0x7f3a848e3000, 61440, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4dc000) = 0x7f3a848e3000
close(3)                                = 0
mprotect(0x7f3a848c1000, 106496, PROT_READ) = 0
mprotect(0x7f3a849c5000, 20480, PROT_READ) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/utils.py", {st_mode=S_IFREG|0644, st_size=21867, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/utils.py", {st_mode=S_IFREG|0644, st_size=21867, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/frontend/tensorflow/__pycache__/utils.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=13180, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372a30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=13180, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\220\263\215ikU\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 13181) = 13180
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging/__init__.cpython-310-x86_64-linux-gnu.so", 0x7ffeca371050) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging/__init__.abi3.so", 0x7ffeca371050) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging/__init__.so", 0x7ffeca371050) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging/__init__.py", {st_mode=S_IFREG|0644, st_size=494, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging/__init__.py", {st_mode=S_IFREG|0644, st_size=494, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging/__pycache__/__init__.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=473, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3713c0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=473, ...}) = 0
read(3, "o\r\r\n\0\0\0\0(z\367h\356\1\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 474) = 473
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bb5ca90 /* 19 entries */, 32768) = 640
getdents64(3, 0x2bb5ca90 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging/version.py", {st_mode=S_IFREG|0644, st_size=16676, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging/version.py", {st_mode=S_IFREG|0644, st_size=16676, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging/__pycache__/version.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=14990, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371c30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=14990, ...}) = 0
read(3, "o\r\r\n\0\0\0\0(z\367h$A\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 14991) = 14990
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging/_structures.py", {st_mode=S_IFREG|0644, st_size=1431, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging/_structures.py", {st_mode=S_IFREG|0644, st_size=1431, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/packaging/__pycache__/_structures.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=2655, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca370e30)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=2655, ...}) = 0
read(3, "o\r\r\n\0\0\0\0(z\367h\227\5\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 2656) = 2655
read(3, "", 1)                          = 0
close(3)                                = 0
brk(0x2bba4000)                         = 0x2bba4000
stat("", 0x7ffeca374890)                = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bb64750 /* 36 entries */, 32768) = 1192
getdents64(3, 0x2bb64750 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python310.zip", 0x7ffeca374890) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python310.zip", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python310.zip", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
getdents64(3, 0x2bb64750 /* 210 entries */, 32768) = 7088
getdents64(3, 0x2bb64750 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bb64750 /* 77 entries */, 32768) = 4840
getdents64(3, 0x2bb64750 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.local/lib/python3.10/site-packages", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.local/lib/python3.10/site-packages", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bb64750 /* 2 entries */, 32768) = 48
getdents64(3, 0x2bb64750 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|S_ISGID|0755, st_size=20480, ...}) = 0
getdents64(3, 0x2bb64750 /* 371 entries */, 32768) = 15120
getdents64(3, 0x2bb64750 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr/Desktop/programming/fly/project/flygym", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/Desktop/programming/fly/project/flygym", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstat(3, {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
getdents64(3, 0x2bb68ab0 /* 19 entries */, 32768) = 616
getdents64(3, 0x2bb68ab0 /* 0 entries */, 32768) = 0
close(3)                                = 0
stat("/home/loganr", {st_mode=S_IFDIR|0700, st_size=4096, ...}) = 0
stat("/home/loganr", {st_mode=S_IFDIR|0700, st_size=4096, ...}) = 0
stat("/home/loganr/intel/openvino_telemetry", 0x7ffeca374630) = -1 ENOENT (No such file or directory)
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures/thread.py", {st_mode=S_IFREG|0644, st_size=8771, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures/thread.py", {st_mode=S_IFREG|0644, st_size=8771, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/concurrent/futures/__pycache__/thread.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=5960, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca373540)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=5960, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367hC\"\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5961) = 5960
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/queue.py", {st_mode=S_IFREG|0644, st_size=11496, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/queue.py", {st_mode=S_IFREG|0644, st_size=11496, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/queue.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=10787, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca372740)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=10787, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h\350,\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 10788) = 10787
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/heapq.py", {st_mode=S_IFREG|0644, st_size=22877, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/heapq.py", {st_mode=S_IFREG|0644, st_size=22877, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/__pycache__/heapq.cpython-310.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=13860, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca371940)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=13860, ...}) = 0
read(3, "o\r\r\n\0\0\0\0\20\272\367h]Y\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 13861) = 13860
read(3, "", 1)                          = 0
close(3)                                = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_heapq.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=104584, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_heapq.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=104584, ...}) = 0
mmap(NULL, 33240, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a84930000
mmap(0x7f3a84931000, 12288, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1000) = 0x7f3a84931000
mmap(0x7f3a84934000, 12288, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x4000) = 0x7f3a84934000
mmap(0x7f3a84937000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x6000) = 0x7f3a84937000
close(3)                                = 0
mprotect(0x7f3a84937000, 4096, PROT_READ) = 0
getcwd("/home/loganr/Desktop/programming/fly/project", 1024) = 45
stat("/home/loganr/Desktop/programming/fly/project", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10", {st_mode=S_IFDIR|S_ISGID|0755, st_size=12288, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_queue.cpython-310-x86_64-linux-gnu.so", {st_mode=S_IFREG|0755, st_size=55568, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/lib-dynload/_queue.cpython-310-x86_64-linux-gnu.so", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=55568, ...}) = 0
mmap(NULL, 21392, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f3a8492a000
mmap(0x7f3a8492c000, 4096, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f3a8492c000
mmap(0x7f3a8492d000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a8492d000
mmap(0x7f3a8492e000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7f3a8492e000
close(3)                                = 0
mprotect(0x7f3a8492e000, 4096, PROT_READ) = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/online", O_RDONLY|O_CLOEXEC) = 3
read(3, "0-15\n", 1024)                 = 5
close(3)                                = 0
stat("/home/loganr", {st_mode=S_IFDIR|0700, st_size=4096, ...}) = 0
stat("/home/loganr/intel/openvino_ga_cid", {st_mode=S_IFREG|0644, st_size=36, ...}) = 0
openat(AT_FDCWD, "/home/loganr/intel/openvino_ga_cid", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=36, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3743f0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
ioctl(3, TCGETS2, 0x7ffeca374230)       = -1 ENOTTY (Inappropriate ioctl for device)
read(3, "e9908b7b-904a-453f-947b-25fcd779"..., 8192) = 36
read(3, "", 8192)                       = 0
close(3)                                = 0
stat("/home/loganr", {st_mode=S_IFDIR|0700, st_size=4096, ...}) = 0
access("/home/loganr/intel/stats", R_OK) = 0
stat("/home/loganr", {st_mode=S_IFDIR|0700, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/intel/stats", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=25, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3745b0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
ioctl(3, TCGETS2, 0x7ffeca3743f0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0644, st_size=25, ...}) = 0
read(3, "{\n    \"usage_count\": 64\n}", 26) = 25
read(3, "", 1)                          = 0
close(3)                                = 0
stat("/home/loganr", {st_mode=S_IFDIR|0700, st_size=4096, ...}) = 0
stat("/home/loganr", {st_mode=S_IFDIR|0700, st_size=4096, ...}) = 0
stat("/home/loganr/intel/stats", {st_mode=S_IFREG|0644, st_size=25, ...}) = 0
stat("/home/loganr", {st_mode=S_IFDIR|0700, st_size=4096, ...}) = 0
access("/home/loganr/intel/stats", W_OK) = 0
stat("/home/loganr", {st_mode=S_IFDIR|0700, st_size=4096, ...}) = 0
openat(AT_FDCWD, "/home/loganr/intel/stats", O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC, 0666) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=0, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca3745b0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
ioctl(3, TCGETS2, 0x7ffeca3743f0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
write(3, "{\n    \"usage_count\": 65\n}", 25) = 25
close(3)                                = 0
stat("/home/loganr", {st_mode=S_IFDIR|0700, st_size=4096, ...}) = 0
stat("/home/loganr", {st_mode=S_IFDIR|0700, st_size=4096, ...}) = 0
stat("/home/loganr", {st_mode=S_IFDIR|0700, st_size=4096, ...}) = 0
access("/home/loganr", W_OK)            = 0
stat("/home/loganr/intel", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/intel", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
stat("/home/loganr/intel", {st_mode=S_IFDIR|0755, st_size=4096, ...}) = 0
access("/home/loganr/intel", W_OK)      = 0
getrandom("\xf5\x9a\xc6\xe0\x77\xd6\x69\xb3\x2d\x06\x57\x39\x7e\xa1\x4c\x0d", 16, 0) = 16
stat("/.dockerenv", 0x7ffeca374e40)     = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/proc/self/cgroup", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0444, st_size=0, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374dc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
read(3, "0::/user.slice/user-1000.slice/u"..., 8192) = 113
read(3, "", 8192)                       = 0
close(3)                                = 0
openat(AT_FDCWD, "/proc/self/mountinfo", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0444, st_size=0, ...}) = 0
ioctl(3, TCGETS2, 0x7ffeca374dc0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(3, 0, SEEK_CUR)                   = 0
read(3, "22 29 0:21 / /proc rw,nosuid,nod"..., 8192) = 3374
read(3, "", 8192)                       = 0
close(3)                                = 0
rt_sigaction(SIGRT_1, {sa_handler=0x7f3a89494070, sa_mask=[], sa_flags=SA_RESTORER|SA_ONSTACK|SA_RESTART|SA_SIGINFO, sa_restorer=0x7f3a8943e4d0}, NULL, 8) = 0
rt_sigprocmask(SIG_UNBLOCK, [RTMIN RT_1], NULL, 8) = 0
mmap(NULL, 8392704, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7f3a83bff000
madvise(0x7f3a83bff000, 4096, MADV_GUARD_INSTALL) = -1 EINVAL (Invalid argument)
mprotect(0x7f3a83bff000, 4096, PROT_NONE) = 0
rt_sigprocmask(SIG_BLOCK, ~[], [], 8)   = 0
clone3({flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM|CLONE_SETTLS|CLONE_PARENT_SETTID|CLONE_CHILD_CLEARTID, child_tid=0x7f3a843ff990, parent_tid=0x7f3a843ff990, exit_signal=0, stack=0x7f3a83bff000, stack_size=0x7fff80, tls=0x7f3a843ff6c0} => {parent_tid=[1266992]}, 88) = 1266992
rt_sigprocmask(SIG_SETMASK, [], NULL, 8) = 0
futex(0x2b5c5d00, FUTEX_WAIT_BITSET_PRIVATE|FUTEX_CLOCK_REALTIME, 0, NULL, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a60, FUTEX_WAIT_BITSET_PRIVATE, 0, {tv_sec=1815620, tv_nsec=99905680}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 0, {tv_sec=1815620, tv_nsec=100267887}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a60, FUTEX_WAKE_PRIVATE, 1)  = 1
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 1
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/telemetry_utils.py", {st_mode=S_IFREG|0644, st_size=4642, ...}) = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 1, {tv_sec=1815620, tv_nsec=100393375}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a64, FUTEX_WAKE_PRIVATE, 1)  = 1
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 1
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/telemetry_utils.py", O_RDONLY|O_CLOEXEC) = 4
futex(0x754a60, FUTEX_WAIT_BITSET_PRIVATE, 4, {tv_sec=1815620, tv_nsec=100488901}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 5, {tv_sec=1815620, tv_nsec=100501962}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a60, FUTEX_WAKE_PRIVATE, 1)  = 1
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 1
fstat(4, {st_mode=S_IFREG|0644, st_size=4642, ...}) = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 7, {tv_sec=1815620, tv_nsec=100560178}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 9, {tv_sec=1815620, tv_nsec=100666171}, FUTEX_BITSET_MATCH_ANY) = -1 EAGAIN (Resource temporarily unavailable)
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a60, FUTEX_WAKE_PRIVATE, 1)  = 1
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 1
ioctl(4, TCGETS2, 0x7ffeca3745d0)       = -1 ENOTTY (Inappropriate ioctl for device)
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 11, {tv_sec=1815620, tv_nsec=100720189}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a60, FUTEX_WAIT_BITSET_PRIVATE, 12, {tv_sec=1815620, tv_nsec=100788285}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 13, {tv_sec=1815620, tv_nsec=100845187}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 15, {tv_sec=1815620, tv_nsec=100903955}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a60, FUTEX_WAKE_PRIVATE, 1)  = 1
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 1
lseek(4, 0, SEEK_CUR)                   = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 17, {tv_sec=1815620, tv_nsec=100964453}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 19, {tv_sec=1815620, tv_nsec=101370617}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a60, FUTEX_WAKE_PRIVATE, 1)  = 1
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 1
read(4, "# Copyright (C) 2018-2025 Intel "..., 4096) = 4096
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 21, {tv_sec=1815620, tv_nsec=101437179}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 23, {tv_sec=1815620, tv_nsec=101482666}, FUTEX_BITSET_MATCH_ANY) = -1 EAGAIN (Resource temporarily unavailable)
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a60, FUTEX_WAKE_PRIVATE, 1)  = 1
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 1
read(4, "ms: command-line parameters dict"..., 8192) = 546
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 25, {tv_sec=1815620, tv_nsec=101577273}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a60, FUTEX_WAKE_PRIVATE, 1)  = 1
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 1
read(4, "", 8192)                       = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 27, {tv_sec=1815620, tv_nsec=101747777}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a60, FUTEX_WAIT_BITSET_PRIVATE, 28, {tv_sec=1815620, tv_nsec=102633101}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 29, {tv_sec=1815620, tv_nsec=102901193}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a60, FUTEX_WAIT_BITSET_PRIVATE, 30, {tv_sec=1815620, tv_nsec=102945869}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 31, {tv_sec=1815620, tv_nsec=103017713}, FUTEX_BITSET_MATCH_ANY) = -1 EAGAIN (Resource temporarily unavailable)
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 35, {tv_sec=1815620, tv_nsec=103072885}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a60, FUTEX_WAIT_BITSET_PRIVATE, 36, {tv_sec=1815620, tv_nsec=103495362}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
futex(0x754a64, FUTEX_WAIT_BITSET_PRIVATE, 41, {tv_sec=1815620, tv_nsec=103706534}, FUTEX_BITSET_MATCH_ANY) = 0
futex(0x754a70, FUTEX_WAKE_PRIVATE, 1)  = 0
close(4)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__init__.py", {st_mode=S_IFREG|0644, st_size=1443, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__init__.py", O_RDONLY|O_CLOEXEC) = 4
fstat(4, {st_mode=S_IFREG|0644, st_size=1443, ...}) = 0
ioctl(4, TCGETS2, 0x7ffeca3745d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(4, 0, SEEK_CUR)                   = 0
read(4, "# Copyright (C) 2018-2025 Intel "..., 4096) = 1443
read(4, "", 8192)                       = 0
close(4)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__init__.py", {st_mode=S_IFREG|0644, st_size=3547, ...}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__init__.py", O_RDONLY|O_CLOEXEC) = 4
fstat(4, {st_mode=S_IFREG|0644, st_size=3547, ...}) = 0
ioctl(4, TCGETS2, 0x7ffeca3745d0)       = -1 ENOTTY (Inappropriate ioctl for device)
lseek(4, 0, SEEK_CUR)                   = 0
read(4, "# -*- coding: utf-8 -*-\n# Copyri"..., 4096) = 3547
read(4, "", 8192)                       = 0
close(4)                                = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__init__.py", {st_mode=S_IFREG|0644, st_size=1443, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__init__.py", {st_mode=S_IFREG|0644, st_size=3547, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/telemetry_utils.py", {st_mode=S_IFREG|0644, st_size=4642, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/__init__.py", {st_mode=S_IFREG|0644, st_size=1443, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/__init__.py", {st_mode=S_IFREG|0644, st_size=3547, ...}) = 0
stat("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/tools/ovc/telemetry_utils.py", {st_mode=S_IFREG|0644, st_size=4642, ...}) = 0
readlink("/home", 0x7ffeca375ff0, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375ff0, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375ff0, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375ff0, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375ff0, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375ff0, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375ff0, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375ff0, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375ff0, 1023) = -1 EINVAL (Invalid argument)
faccessat2(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/", F_OK, AT_EACCESS) = 0
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375ff0, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375ff0, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino.so.2540", 0x7ffeca375ff0, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/openvino-2025.4.0/plugins.xml", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/plugins.xml", O_RDONLY) = -1 ENOENT (No such file or directory)
readlink("/home", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
faccessat2(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/", F_OK, AT_EACCESS) = 0
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino.so.2540", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/openvino-2025.4.0/libopenvino_auto_plugin.so", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 565745
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 565745
close(4)                                = 0
readlink("/home", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
faccessat2(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/", F_OK, AT_EACCESS) = 0
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino.so.2540", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/openvino-2025.4.0/libopenvino_auto_batch_plugin.so", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_batch_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 253297
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_batch_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 253297
close(4)                                = 0
readlink("/home", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
faccessat2(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/", F_OK, AT_EACCESS) = 0
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino.so.2540", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/openvino-2025.4.0/libopenvino_intel_cpu_plugin.so", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_cpu_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 52356657
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_cpu_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 52356657
close(4)                                = 0
readlink("/home", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
faccessat2(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/", F_OK, AT_EACCESS) = 0
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino.so.2540", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/openvino-2025.4.0/libopenvino_intel_gpu_plugin.so", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_gpu_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 37232649
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_gpu_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 37232649
close(4)                                = 0
readlink("/home", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
faccessat2(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/", F_OK, AT_EACCESS) = 0
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino.so.2540", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/openvino-2025.4.0/libopenvino_hetero_plugin.so", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_hetero_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 484329
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_hetero_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 484329
close(4)                                = 0
readlink("/home", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
faccessat2(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/", F_OK, AT_EACCESS) = 0
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino.so.2540", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/openvino-2025.4.0/libopenvino_auto_plugin.so", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 565745
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 565745
close(4)                                = 0
readlink("/home", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
faccessat2(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/", F_OK, AT_EACCESS) = 0
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino.so.2540", 0x7ffeca375910, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/openvino-2025.4.0/libopenvino_intel_npu_plugin.so", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_npu_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 5884289
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_npu_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 5884289
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_plugin.so", O_RDONLY|O_CLOEXEC) = 4
read(4, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0000.\1\0\0\0\0\0"..., 832) = 832
fstat(4, {st_mode=S_IFREG|0644, st_size=565745, ...}) = 0
mmap(NULL, 569840, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7f3a83a73000
mmap(0x7f3a83a85000, 360448, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x12000) = 0x7f3a83a85000
mmap(0x7f3a83add000, 90112, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x6a000) = 0x7f3a83add000
mmap(0x7f3a83af3000, 20480, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x7f000) = 0x7f3a83af3000
mmap(0x7f3a83af8000, 28672, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x84000) = 0x7f3a83af8000
close(4)                                = 0
mprotect(0x7f3a83af3000, 16384, PROT_READ) = 0
readlink("/home", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_plugin.so", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 565745
close(4)                                = 0
futex(0x7f3a8963d210, FUTEX_WAKE_PRIVATE, 2147483647) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_batch_plugin.so", O_RDONLY|O_CLOEXEC) = 4
read(4, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\320\333\0\0\0\0\0\0"..., 832) = 832
fstat(4, {st_mode=S_IFREG|0644, st_size=253297, ...}) = 0
mmap(NULL, 257392, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7f3a83a34000
mprotect(0x7f3a83a41000, 184320, PROT_NONE) = 0
mmap(0x7f3a83a41000, 126976, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0xd000) = 0x7f3a83a41000
mmap(0x7f3a83a60000, 40960, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x2c000) = 0x7f3a83a60000
mmap(0x7f3a83a6b000, 12288, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x36000) = 0x7f3a83a6b000
mmap(0x7f3a83a6e000, 20480, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x39000) = 0x7f3a83a6e000
close(4)                                = 0
mprotect(0x7f3a83a6b000, 8192, PROT_READ) = 0
readlink("/home", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_batch_plugin.so", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_batch_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 253297
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_cpu_plugin.so", O_RDONLY|O_CLOEXEC) = 4
read(4, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0`\200*\0\0\0\0\0"..., 832) = 832
fstat(4, {st_mode=S_IFREG|0644, st_size=52356657, ...}) = 0
mmap(NULL, 52586032, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7f3a80800000
mprotect(0x7f3a80aa3000, 49700864, PROT_NONE) = 0
mmap(0x7f3a80aa3000, 41906176, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x2a3000) = 0x7f3a80aa3000
mmap(0x7f3a8329a000, 6627328, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x2a9a000) = 0x7f3a8329a000
mmap(0x7f3a838ec000, 942080, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x30eb000) = 0x7f3a838ec000
mmap(0x7f3a839d2000, 224904, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a839d2000
mmap(0x7f3a83a09000, 122880, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x31d1000) = 0x7f3a83a09000
close(4)                                = 0
mprotect(0x7f3a838ec000, 929792, PROT_READ) = 0
openat(AT_FDCWD, "/sys/devices/system/node/node0/cpulist", O_RDONLY) = 4
read(4, "0-15\n", 8191)                 = 5
close(4)                                = 0
openat(AT_FDCWD, "/sys/devices/system/node/node1/cpulist", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/sys/devices/system/cpu/possible", O_RDONLY) = 4
read(4, "0-15\n", 8191)                 = 5
openat(AT_FDCWD, "/sys/devices/system/cpu/online", O_RDONLY) = 5
read(5, "0-15\n", 8191)                 = 5
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu0/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "0-1\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu0/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "0-1\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu1/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "0-1\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu1/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "0-1\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu1/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu2/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "2-3\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu2/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "2-3\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu2/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu3/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "2-3\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu3/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "2-3\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu3/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu4/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "4-5\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu4/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "4-5\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu4/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu5/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "4-5\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu5/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "4-5\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu5/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu6/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "6-7\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu6/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "6-7\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu6/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu7/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "6-7\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu7/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "6-7\n", 8191)                  = 4
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu7/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu8/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "8\n", 8191)                    = 2
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu8/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "8-11\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu8/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu9/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "9\n", 8191)                    = 2
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu9/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "8-11\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu9/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu10/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "10\n", 8191)                   = 3
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu10/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "8-11\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu10/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu11/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "11\n", 8191)                   = 3
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu11/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "8-11\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu11/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu12/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "12\n", 8191)                   = 3
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu12/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "12-15\n", 8191)                = 6
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu12/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu13/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "13\n", 8191)                   = 3
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu13/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "12-15\n", 8191)                = 6
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu13/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu14/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "14\n", 8191)                   = 3
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu14/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "12-15\n", 8191)                = 6
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu14/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu15/cache/index0/shared_cpu_list", O_RDONLY) = 6
read(6, "15\n", 8191)                   = 3
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu15/cache/index2/shared_cpu_list", O_RDONLY) = 6
read(6, "12-15\n", 8191)                = 6
close(6)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/cpu15/cache/index3/shared_cpu_list", O_RDONLY) = 6
read(6, "0-15\n", 8191)                 = 5
close(6)                                = 0
close(5)                                = 0
close(4)                                = 0
getpid()                                = 1266982
sched_getaffinity(1266982, 8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]) = 8
mmap(NULL, 11824, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7f3a88b19000
sigaltstack({ss_sp=0x7f3a88b19000, ss_flags=0, ss_size=11824}, {ss_sp=NULL, ss_flags=SS_DISABLE, ss_size=0}) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libirml.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libirml.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/libirml.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 4
fstat(4, {st_mode=S_IFREG|0644, st_size=243639, ...}) = 0
mmap(NULL, 243639, PROT_READ, MAP_PRIVATE, 4, 0) = 0x7f3a807c4000
close(4)                                = 0
openat(AT_FDCWD, "/usr/lib/libirml.so.1", O_RDONLY|O_CLOEXEC) = 4
read(4, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(4, {st_mode=S_IFREG|0755, st_size=14288, ...}) = 0
close(4)                                = 0
munmap(0x7f3a807c4000, 243639)          = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libirml.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/sys/devices/system/cpu/online", O_RDONLY|O_CLOEXEC) = 4
read(4, "0-15\n", 1024)                 = 5
close(4)                                = 0
getpid()                                = 1266982
sched_getaffinity(1266982, 128, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]) = 8
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libiomp5.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libiomp5.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/libiomp5.so", O_RDONLY|O_CLOEXEC) = 4
read(4, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(4, {st_mode=S_IFREG|0755, st_size=1462104, ...}) = 0
close(4)                                = 0
sched_getaffinity(0, 128, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]) = 8
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libtbbbind_2_5.so.3", O_RDONLY|O_CLOEXEC) = 4
read(4, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0000#\0\0\0\0\0\0"..., 832) = 832
fstat(4, {st_mode=S_IFREG|0644, st_size=31361, ...}) = 0
mmap(NULL, 35456, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7f3a8491b000
mmap(0x7f3a8491d000, 12288, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x2000) = 0x7f3a8491d000
mmap(0x7f3a84920000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x5000) = 0x7f3a84920000
mmap(0x7f3a84921000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x5000) = 0x7f3a84921000
mmap(0x7f3a84923000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x7000) = 0x7f3a84923000
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/../openvino/libs/libhwloc.so.15", O_RDONLY|O_CLOEXEC) = 4
read(4, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\360k\0\0\0\0\0\0"..., 832) = 832
fstat(4, {st_mode=S_IFREG|0644, st_size=471657, ...}) = 0
mmap(NULL, 475752, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7f3a8078b000
mprotect(0x7f3a80791000, 442368, PROT_NONE) = 0
mmap(0x7f3a80791000, 348160, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x6000) = 0x7f3a80791000
mmap(0x7f3a807e6000, 81920, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x5b000) = 0x7f3a807e6000
mmap(0x7f3a807fb000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x6f000) = 0x7f3a807fb000
mmap(0x7f3a807fd000, 12288, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x71000) = 0x7f3a807fd000
close(4)                                = 0
mprotect(0x7f3a807fb000, 4096, PROT_READ) = 0
mprotect(0x7f3a84921000, 4096, PROT_READ) = 0
faccessat2(-1, "/sys/devices/system/cpu", R_OK|X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu0/topology/package_cpus", R_OK, 0) = 0
uname({sysname="Linux", nodename="loptop", ...}) = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/online", O_RDONLY|O_CLOEXEC) = 4
read(4, "0-15\n", 1024)                 = 5
close(4)                                = 0
openat(-1, "/proc/cpuinfo", O_RDONLY)   = 4
fcntl(4, F_GETFL)                       = 0x8000 (flags O_RDONLY|O_LARGEFILE)
fstat(4, {st_mode=S_IFREG|0444, st_size=0, ...}) = 0
read(4, "processor\t: 0\nvendor_id\t: Genuin"..., 1024) = 1024
read(4, "ck_detect user_shstk avx_vnni dt"..., 1024) = 1024
read(4, "l\t: 32\nwp\t\t: yes\nflags\t\t: fpu vm"..., 1024) = 1024
read(4, "pt_x_only ept_ad ept_1gb flexpri"..., 1024) = 1024
read(4, "x smx est tm2 ssse3 sdbg fma cx1"..., 1024) = 1024
read(4, "ress sizes\t: 39 bits physical, 4"..., 1024) = 1024
read(4, "shopt clwb intel_pt sha_ni xsave"..., 1024) = 1024
read(4, "\t\t: 16\ninitial apicid\t: 16\nfpu\t\t"..., 1024) = 1024
read(4, "exit_to_user\nvmx flags\t: vnmi pr"..., 1024) = 1024
read(4, " cpuid aperfmperf tsc_known_freq"..., 1024) = 1024
read(4, "i vmscape\nbogomips\t: 4993.00\nclf"..., 1024) = 1024
read(4, "fsgsbase tsc_adjust bmi1 avx2 sm"..., 1024) = 1024
read(4, " 18432 KB\nphysical id\t: 0\nsiblin"..., 1024) = 1024
read(4, "r64b fsrm md_clear serialize arc"..., 1024) = 1024
read(4, "m constant_tsc art arch_perfmon "..., 1024) = 1024
read(4, "e\nbugs\t\t: spectre_v1 spectre_v2 "..., 1024) = 1024
read(4, " ssbd ibrs ibpb stibp ibrs_enhan"..., 1024) = 1024
read(4, "(TM) i7-1260P\nstepping\t: 3\nmicro"..., 1024) = 1024
read(4, "_req hfi vnmi umip pku ospke wai"..., 1024) = 1024
read(4, "36 clflush dts acpi mmx fxsr sse"..., 1024) = 1024
read(4, "st vapic_reg vid ple shadow_vmcs"..., 1024) = 1024
read(4, "dline_timer aes xsave avx f16c r"..., 1024) = 1024
read(4, "\t: GenuineIntel\ncpu family\t: 6\nm"..., 1024) = 1024
read(4, "avx_vnni dtherm ida arat pln pts"..., 1024) = 1024
read(4, "flags\t\t: fpu vme de pse tsc msr "..., 1024) = 1024
read(4, "ept_1gb flexpriority apicv tsc_o"..., 1024) = 1024
read(4, " ssse3 sdbg fma cx16 xtpr pdcm p"..., 1024) = 1024
read(4, "39 bits physical, 48 bits virtua"..., 1024) = 53
read(4, "", 1024)                       = 0
close(4)                                = 0
openat(AT_FDCWD, "/proc/mounts", O_RDONLY|O_CLOEXEC) = 4
fstat(4, {st_mode=S_IFREG|0444, st_size=0, ...}) = 0
read(4, "proc /proc proc rw,nosuid,nodev,"..., 1024) = 1024
openat(-1, "/sys/fs/cgroup/cgroup.controllers", O_RDONLY) = 5
read(5, "cpuset cpu io memory hugetlb pid"..., 1023) = 44
close(5)                                = 0
lseek(4, 0, SEEK_CUR)                   = 1024
lseek(4, 719, SEEK_SET)                 = 719
close(4)                                = 0
openat(-1, "/proc/self/cpuset", O_RDONLY) = 4
read(4, "/\n", 127)                     = 2
close(4)                                = 0
openat(-1, "/sys/fs/cgroup//cpuset.cpus.effective", O_RDONLY) = 4
read(4, "0-15\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/fs/cgroup//cpuset.mems.effective", O_RDONLY) = 4
read(4, "0\n", 4097)                    = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/online", O_RDONLY) = 4
read(4, "0-15\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu", O_RDONLY|O_DIRECTORY) = 4
fstat(4, {st_mode=S_IFDIR|0755, st_size=0, ...}) = 0
fcntl(4, F_GETFL)                       = 0x18000 (flags O_RDONLY|O_LARGEFILE|O_DIRECTORY)
fcntl(4, F_SETFD, FD_CLOEXEC)           = 0
getdents64(4, 0x2bb8c730 /* 38 entries */, 32768) = 1152
faccessat2(-1, "/sys/devices/system/cpu/cpu11/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu9/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu7/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu5/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu3/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu14/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu1/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu12/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu10/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu8/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu6/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu4/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu15/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu2/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu13/topology", X_OK, 0) = 0
faccessat2(-1, "/sys/devices/system/cpu/cpu0/topology", X_OK, 0) = 0
getdents64(4, 0x2bb8c730 /* 0 entries */, 32768) = 0
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/topology/core_cpus", O_RDONLY) = 4
read(4, "0003\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/topology/core_id", O_RDONLY) = 4
read(4, "0\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/topology/cluster_cpus", O_RDONLY) = 4
read(4, "0003\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/topology/die_cpus", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/topology/package_cpus", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/topology/physical_package_id", O_RDONLY) = 4
read(4, "0\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/topology/cluster_id", O_RDONLY) = 4
read(4, "0\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "0003\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index0/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index0/type", O_RDONLY) = 4
read(4, "Data\n", 19)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index0/id", O_RDONLY) = 4
read(4, "0\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index0/size", O_RDONLY) = 4
read(4, "48K\n", 10)                    = 4
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index0/number_of_sets", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index0/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "0003\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index1/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index1/type", O_RDONLY) = 4
read(4, "Instruction\n", 19)            = 12
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index1/id", O_RDONLY) = 4
read(4, "0\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "0003\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index2/level", O_RDONLY) = 4
read(4, "2\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index2/type", O_RDONLY) = 4
read(4, "Unified\n", 19)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index2/id", O_RDONLY) = 4
read(4, "0\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index2/size", O_RDONLY) = 4
read(4, "1280K\n", 10)                  = 6
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index2/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index2/number_of_sets", O_RDONLY) = 4
read(4, "2048\n", 10)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index2/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index3/level", O_RDONLY) = 4
read(4, "3\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index3/type", O_RDONLY) = 4
read(4, "Unified\n", 19)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index3/id", O_RDONLY) = 4
read(4, "0\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index3/size", O_RDONLY) = 4
read(4, "18432K\n", 10)                 = 7
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index3/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index3/number_of_sets", O_RDONLY) = 4
read(4, "24576\n", 10)                  = 6
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index3/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu0/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu1/topology/core_cpus", O_RDONLY) = 4
read(4, "0003\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu1/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "0003\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu1/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "0003\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu1/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "0003\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu1/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu1/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu1/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu1/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu1/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu1/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu1/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu2/topology/core_cpus", O_RDONLY) = 4
read(4, "000c\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/topology/core_id", O_RDONLY) = 4
read(4, "4\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/topology/cluster_cpus", O_RDONLY) = 4
read(4, "000c\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/topology/die_cpus", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/topology/cluster_id", O_RDONLY) = 4
read(4, "8\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "000c\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index0/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index0/type", O_RDONLY) = 4
read(4, "Data\n", 19)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index0/id", O_RDONLY) = 4
read(4, "4\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index0/size", O_RDONLY) = 4
read(4, "48K\n", 10)                    = 4
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index0/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index0/number_of_sets", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index0/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "000c\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index1/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index1/type", O_RDONLY) = 4
read(4, "Instruction\n", 19)            = 12
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index1/id", O_RDONLY) = 4
read(4, "4\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "000c\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index2/level", O_RDONLY) = 4
read(4, "2\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index2/type", O_RDONLY) = 4
read(4, "Unified\n", 19)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index2/id", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index2/size", O_RDONLY) = 4
read(4, "1280K\n", 10)                  = 6
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index2/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index2/number_of_sets", O_RDONLY) = 4
read(4, "2048\n", 10)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index2/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu2/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu3/topology/core_cpus", O_RDONLY) = 4
read(4, "000c\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu3/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "000c\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu3/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "000c\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu3/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "000c\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu3/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu3/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu3/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu3/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu3/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu3/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu3/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu4/topology/core_cpus", O_RDONLY) = 4
read(4, "0030\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/topology/core_id", O_RDONLY) = 4
read(4, "8\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/topology/cluster_cpus", O_RDONLY) = 4
read(4, "0030\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/topology/die_cpus", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/topology/cluster_id", O_RDONLY) = 4
read(4, "16\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "0030\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index0/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index0/type", O_RDONLY) = 4
read(4, "Data\n", 19)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index0/id", O_RDONLY) = 4
read(4, "8\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index0/size", O_RDONLY) = 4
read(4, "48K\n", 10)                    = 4
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index0/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index0/number_of_sets", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index0/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "0030\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index1/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index1/type", O_RDONLY) = 4
read(4, "Instruction\n", 19)            = 12
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index1/id", O_RDONLY) = 4
read(4, "8\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "0030\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index2/level", O_RDONLY) = 4
read(4, "2\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index2/type", O_RDONLY) = 4
read(4, "Unified\n", 19)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index2/id", O_RDONLY) = 4
read(4, "2\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index2/size", O_RDONLY) = 4
read(4, "1280K\n", 10)                  = 6
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index2/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index2/number_of_sets", O_RDONLY) = 4
read(4, "2048\n", 10)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index2/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu4/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu5/topology/core_cpus", O_RDONLY) = 4
read(4, "0030\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu5/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "0030\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu5/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "0030\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu5/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "0030\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu5/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu5/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu5/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu5/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu5/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu5/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu5/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu6/topology/core_cpus", O_RDONLY) = 4
read(4, "00c0\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/topology/core_id", O_RDONLY) = 4
read(4, "12\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/topology/cluster_cpus", O_RDONLY) = 4
read(4, "00c0\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/topology/die_cpus", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/topology/cluster_id", O_RDONLY) = 4
read(4, "24\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "00c0\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index0/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index0/type", O_RDONLY) = 4
read(4, "Data\n", 19)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index0/id", O_RDONLY) = 4
read(4, "12\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index0/size", O_RDONLY) = 4
read(4, "48K\n", 10)                    = 4
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index0/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index0/number_of_sets", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index0/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "00c0\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index1/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index1/type", O_RDONLY) = 4
read(4, "Instruction\n", 19)            = 12
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index1/id", O_RDONLY) = 4
read(4, "12\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "00c0\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index2/level", O_RDONLY) = 4
read(4, "2\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index2/type", O_RDONLY) = 4
read(4, "Unified\n", 19)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index2/id", O_RDONLY) = 4
read(4, "3\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index2/size", O_RDONLY) = 4
read(4, "1280K\n", 10)                  = 6
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index2/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index2/number_of_sets", O_RDONLY) = 4
read(4, "2048\n", 10)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index2/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu6/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu7/topology/core_cpus", O_RDONLY) = 4
read(4, "00c0\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu7/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "00c0\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu7/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "00c0\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu7/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "00c0\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu7/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu7/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu7/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu7/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu7/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu7/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu7/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu8/topology/core_cpus", O_RDONLY) = 4
read(4, "0100\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/topology/core_id", O_RDONLY) = 4
read(4, "16\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/topology/cluster_cpus", O_RDONLY) = 4
read(4, "0f00\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/topology/die_cpus", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/topology/cluster_id", O_RDONLY) = 4
read(4, "32\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "0100\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index0/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index0/type", O_RDONLY) = 4
read(4, "Data\n", 19)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index0/id", O_RDONLY) = 4
read(4, "16\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index0/size", O_RDONLY) = 4
read(4, "32K\n", 10)                    = 4
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index0/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index0/number_of_sets", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index0/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "0100\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index1/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index1/type", O_RDONLY) = 4
read(4, "Instruction\n", 19)            = 12
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index1/id", O_RDONLY) = 4
read(4, "16\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "0f00\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index2/level", O_RDONLY) = 4
read(4, "2\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index2/type", O_RDONLY) = 4
read(4, "Unified\n", 19)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index2/id", O_RDONLY) = 4
read(4, "4\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index2/size", O_RDONLY) = 4
read(4, "2048K\n", 10)                  = 6
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index2/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index2/number_of_sets", O_RDONLY) = 4
read(4, "2048\n", 10)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index2/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu8/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu9/topology/core_cpus", O_RDONLY) = 4
read(4, "0200\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/topology/core_id", O_RDONLY) = 4
read(4, "17\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/topology/cluster_cpus", O_RDONLY) = 4
read(4, "0f00\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "0200\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index0/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index0/type", O_RDONLY) = 4
read(4, "Data\n", 19)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index0/id", O_RDONLY) = 4
read(4, "17\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index0/size", O_RDONLY) = 4
read(4, "32K\n", 10)                    = 4
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index0/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index0/number_of_sets", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index0/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "0200\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index1/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index1/type", O_RDONLY) = 4
read(4, "Instruction\n", 19)            = 12
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index1/id", O_RDONLY) = 4
read(4, "17\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "0f00\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu9/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu10/topology/core_cpus", O_RDONLY) = 4
read(4, "0400\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/topology/core_id", O_RDONLY) = 4
read(4, "18\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/topology/cluster_cpus", O_RDONLY) = 4
read(4, "0f00\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "0400\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index0/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index0/type", O_RDONLY) = 4
read(4, "Data\n", 19)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index0/id", O_RDONLY) = 4
read(4, "18\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index0/size", O_RDONLY) = 4
read(4, "32K\n", 10)                    = 4
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index0/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index0/number_of_sets", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index0/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "0400\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index1/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index1/type", O_RDONLY) = 4
read(4, "Instruction\n", 19)            = 12
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index1/id", O_RDONLY) = 4
read(4, "18\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "0f00\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu10/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu11/topology/core_cpus", O_RDONLY) = 4
read(4, "0800\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/topology/core_id", O_RDONLY) = 4
read(4, "19\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/topology/cluster_cpus", O_RDONLY) = 4
read(4, "0f00\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "0800\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index0/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index0/type", O_RDONLY) = 4
read(4, "Data\n", 19)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index0/id", O_RDONLY) = 4
read(4, "19\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index0/size", O_RDONLY) = 4
read(4, "32K\n", 10)                    = 4
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index0/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index0/number_of_sets", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index0/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "0800\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index1/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index1/type", O_RDONLY) = 4
read(4, "Instruction\n", 19)            = 12
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index1/id", O_RDONLY) = 4
read(4, "19\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "0f00\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu11/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu12/topology/core_cpus", O_RDONLY) = 4
read(4, "1000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/topology/core_id", O_RDONLY) = 4
read(4, "20\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/topology/cluster_cpus", O_RDONLY) = 4
read(4, "f000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/topology/die_cpus", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/topology/cluster_id", O_RDONLY) = 4
read(4, "40\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "1000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index0/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index0/type", O_RDONLY) = 4
read(4, "Data\n", 19)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index0/id", O_RDONLY) = 4
read(4, "20\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index0/size", O_RDONLY) = 4
read(4, "32K\n", 10)                    = 4
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index0/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index0/number_of_sets", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index0/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "1000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index1/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index1/type", O_RDONLY) = 4
read(4, "Instruction\n", 19)            = 12
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index1/id", O_RDONLY) = 4
read(4, "20\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "f000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index2/level", O_RDONLY) = 4
read(4, "2\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index2/type", O_RDONLY) = 4
read(4, "Unified\n", 19)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index2/id", O_RDONLY) = 4
read(4, "5\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index2/size", O_RDONLY) = 4
read(4, "2048K\n", 10)                  = 6
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index2/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index2/number_of_sets", O_RDONLY) = 4
read(4, "2048\n", 10)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index2/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu12/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu13/topology/core_cpus", O_RDONLY) = 4
read(4, "2000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/topology/core_id", O_RDONLY) = 4
read(4, "21\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/topology/cluster_cpus", O_RDONLY) = 4
read(4, "f000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "2000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index0/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index0/type", O_RDONLY) = 4
read(4, "Data\n", 19)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index0/id", O_RDONLY) = 4
read(4, "21\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index0/size", O_RDONLY) = 4
read(4, "32K\n", 10)                    = 4
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index0/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index0/number_of_sets", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index0/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "2000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index1/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index1/type", O_RDONLY) = 4
read(4, "Instruction\n", 19)            = 12
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index1/id", O_RDONLY) = 4
read(4, "21\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "f000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu13/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu14/topology/core_cpus", O_RDONLY) = 4
read(4, "4000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/topology/core_id", O_RDONLY) = 4
read(4, "22\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/topology/cluster_cpus", O_RDONLY) = 4
read(4, "f000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "4000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index0/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index0/type", O_RDONLY) = 4
read(4, "Data\n", 19)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index0/id", O_RDONLY) = 4
read(4, "22\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index0/size", O_RDONLY) = 4
read(4, "32K\n", 10)                    = 4
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index0/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index0/number_of_sets", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index0/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "4000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index1/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index1/type", O_RDONLY) = 4
read(4, "Instruction\n", 19)            = 12
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index1/id", O_RDONLY) = 4
read(4, "22\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "f000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu14/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu15/topology/core_cpus", O_RDONLY) = 4
read(4, "8000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/topology/core_id", O_RDONLY) = 4
read(4, "23\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/topology/cluster_cpus", O_RDONLY) = 4
read(4, "f000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index0/shared_cpu_map", O_RDONLY) = 4
read(4, "8000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index0/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index0/type", O_RDONLY) = 4
read(4, "Data\n", 19)                   = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index0/id", O_RDONLY) = 4
read(4, "23\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index0/size", O_RDONLY) = 4
read(4, "32K\n", 10)                    = 4
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index0/coherency_line_size", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index0/number_of_sets", O_RDONLY) = 4
read(4, "64\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index0/physical_line_partition", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index1/shared_cpu_map", O_RDONLY) = 4
read(4, "8000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index1/level", O_RDONLY) = 4
read(4, "1\n", 10)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index1/type", O_RDONLY) = 4
read(4, "Instruction\n", 19)            = 12
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index1/id", O_RDONLY) = 4
read(4, "23\n", 10)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index2/shared_cpu_map", O_RDONLY) = 4
read(4, "f000\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index3/shared_cpu_map", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index4/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index5/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index6/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index7/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index8/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu15/cache/index9/shared_cpu_map", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "4700000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "2100000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "4700000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu1/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "2100000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "4700000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu2/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "2100000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "4700000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu3/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "2100000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "4700000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu4/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "2100000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "4700000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu5/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "2100000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "4700000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu6/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "2100000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "4700000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu7/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "2100000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "3400000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu8/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "1500000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "3400000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu9/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "1500000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "3400000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu10/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "1500000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "3400000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu11/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "1500000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "3400000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu12/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "1500000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "3400000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu13/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "1500000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "3400000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu14/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "1500000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cpufreq/cpuinfo_max_freq", O_RDONLY) = 4
read(4, "3400000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu15/cpufreq/base_frequency", O_RDONLY) = 4
read(4, "1500000\n", 10)                = 8
close(4)                                = 0
openat(-1, "/sys/devices/system/cpu/cpu0/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu1/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu2/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu3/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu4/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu5/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu6/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu7/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu8/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu9/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu10/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu11/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu12/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu13/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu14/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/cpu/cpu15/cpu_capacity", O_RDONLY) = -1 ENOENT (No such file or directory)
newfstatat(-1, "/sys/kernel/mm/hugepages", {st_mode=S_IFDIR|0755, st_size=0, ...}, 0) = 0
openat(-1, "/proc/meminfo", O_RDONLY)   = 4
read(4, "MemTotal:       65551164 kB\nMemF"..., 4095) = 1587
close(4)                                = 0
openat(-1, "/sys/kernel/mm/hugepages", O_RDONLY|O_DIRECTORY) = 4
fstat(4, {st_mode=S_IFDIR|0755, st_size=0, ...}) = 0
fcntl(4, F_GETFL)                       = 0x18000 (flags O_RDONLY|O_LARGEFILE|O_DIRECTORY)
fcntl(4, F_SETFD, FD_CLOEXEC)           = 0
getdents64(4, 0x2bb8da50 /* 4 entries */, 32768) = 128
openat(-1, "/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages", O_RDONLY) = 5
read(5, "0\n", 63)                      = 2
close(5)                                = 0
openat(-1, "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages", O_RDONLY) = 5
read(5, "0\n", 63)                      = 2
close(5)                                = 0
getdents64(4, 0x2bb8da50 /* 0 entries */, 32768) = 0
close(4)                                = 0
faccessat2(-1, "/sys/devices/system/node", R_OK|X_OK, 0) = 0
openat(-1, "/sys/devices/system/node/online", O_RDONLY) = 4
read(4, "0\n", 4097)                    = 2
close(4)                                = 0
openat(-1, "/sys/devices/system/node/node0/cpumap", O_RDONLY) = 4
read(4, "ffff\n", 4097)                 = 5
close(4)                                = 0
newfstatat(-1, "/sys/devices/system/node/node0/hugepages", {st_mode=S_IFDIR|0755, st_size=0, ...}, 0) = 0
openat(-1, "/sys/devices/system/node/node0/meminfo", O_RDONLY) = 4
read(4, "Node 0 MemTotal:       65551164 "..., 4095) = 1280
close(4)                                = 0
openat(-1, "/sys/devices/system/node/node0/hugepages", O_RDONLY|O_DIRECTORY) = 4
fstat(4, {st_mode=S_IFDIR|0755, st_size=0, ...}) = 0
fcntl(4, F_GETFL)                       = 0x18000 (flags O_RDONLY|O_LARGEFILE|O_DIRECTORY)
fcntl(4, F_SETFD, FD_CLOEXEC)           = 0
getdents64(4, 0x2bb8da90 /* 4 entries */, 32768) = 128
openat(-1, "/sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages", O_RDONLY) = 5
read(5, "0\n", 63)                      = 2
close(5)                                = 0
openat(-1, "/sys/devices/system/node/node0/hugepages/hugepages-1048576kB/nr_hugepages", O_RDONLY) = 5
read(5, "0\n", 63)                      = 2
close(5)                                = 0
getdents64(4, 0x2bb8da90 /* 0 entries */, 32768) = 0
close(4)                                = 0
openat(-1, "/proc/driver/nvidia/gpus", O_RDONLY|O_DIRECTORY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/bus/dax/devices/", O_RDONLY|O_DIRECTORY) = 4
fstat(4, {st_mode=S_IFDIR|0755, st_size=0, ...}) = 0
fcntl(4, F_GETFL)                       = 0x18000 (flags O_RDONLY|O_LARGEFILE|O_DIRECTORY)
fcntl(4, F_SETFD, FD_CLOEXEC)           = 0
getdents64(4, 0x2bb8da90 /* 2 entries */, 32768) = 48
newfstatat(-1, "/sys/bus/dax/drivers/kmem/.", 0x7ffeca375660, 0) = -1 ENOENT (No such file or directory)
newfstatat(-1, "/sys/bus/dax/drivers/kmem/..", 0x7ffeca375660, 0) = -1 ENOENT (No such file or directory)
getdents64(4, 0x2bb8da90 /* 0 entries */, 32768) = 0
close(4)                                = 0
openat(-1, "/sys/devices/system/node/node0/access1/initiators", O_RDONLY|O_DIRECTORY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/node/node0/access0/initiators", O_RDONLY|O_DIRECTORY) = -1 ENOENT (No such file or directory)
faccessat2(-1, "/sys/devices/system/node/node0/access1/initiators", X_OK, 0) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/node/node0/access0/initiators/read_bandwidth", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/node/node0/access0/initiators/write_bandwidth", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/node/node0/access0/initiators/read_latency", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/system/node/node0/access0/initiators/write_latency", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(-1, "/sys/devices/virtual/dmi/id", O_RDONLY|O_DIRECTORY) = 4
fstat(4, {st_mode=S_IFDIR|0755, st_size=0, ...}) = 0
fcntl(4, F_GETFL)                       = 0x18000 (flags O_RDONLY|O_LARGEFILE|O_DIRECTORY)
fcntl(4, F_SETFD, FD_CLOEXEC)           = 0
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/product_name", O_RDONLY) = 4
read(4, "Laptop (12th Gen Intel Core)\n", 63) = 29
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/product_version", O_RDONLY) = 4
read(4, "A6\n", 63)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/product_serial", O_RDONLY) = -1 EACCES (Permission denied)
openat(-1, "/sys/devices/virtual/dmi/id/product_uuid", O_RDONLY) = -1 EACCES (Permission denied)
openat(-1, "/sys/devices/virtual/dmi/id/board_vendor", O_RDONLY) = 4
read(4, "Framework\n", 63)              = 10
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/board_name", O_RDONLY) = 4
read(4, "FRANMACP06\n", 63)             = 11
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/board_version", O_RDONLY) = 4
read(4, "A6\n", 63)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/board_serial", O_RDONLY) = -1 EACCES (Permission denied)
openat(-1, "/sys/devices/virtual/dmi/id/board_asset_tag", O_RDONLY) = 4
read(4, "*\n", 63)                      = 2
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/chassis_vendor", O_RDONLY) = 4
read(4, "Framework\n", 63)              = 10
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/chassis_type", O_RDONLY) = 4
read(4, "10\n", 63)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/chassis_version", O_RDONLY) = 4
read(4, "A6\n", 63)                     = 3
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/chassis_serial", O_RDONLY) = -1 EACCES (Permission denied)
openat(-1, "/sys/devices/virtual/dmi/id/chassis_asset_tag", O_RDONLY) = 4
read(4, "FRANDACPA63053000G\n", 63)     = 19
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/bios_vendor", O_RDONLY) = 4
read(4, "INSYDE Corp.\n", 63)           = 13
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/bios_version", O_RDONLY) = 4
read(4, "03.06\n", 63)                  = 6
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/bios_date", O_RDONLY) = 4
read(4, "11/10/2022\n", 63)             = 11
close(4)                                = 0
openat(-1, "/sys/devices/virtual/dmi/id/sys_vendor", O_RDONLY) = 4
read(4, "Framework\n", 63)              = 10
close(4)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu/possible", O_RDONLY|O_CLOEXEC) = 4
read(4, "0-15\n", 1024)                 = 5
close(4)                                = 0
openat(AT_FDCWD, "/proc/self/task", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 4
fstat(4, {st_mode=S_IFDIR|0555, st_size=0, ...}) = 0
fstat(4, {st_mode=S_IFDIR|0555, st_size=0, ...}) = 0
lseek(4, 0, SEEK_SET)                   = 0
getdents64(4, 0x2bb8e110 /* 4 entries */, 32768) = 112
getdents64(4, 0x2bb8e110 /* 0 entries */, 32768) = 0
openat(-1, "/sys/devices/system/cpu/possible", O_RDONLY) = 5
read(5, "0-15\n", 4097)                 = 5
close(5)                                = 0
sched_getaffinity(0, 8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]) = 8
sched_getaffinity(1266982, 8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]) = 8
sched_getaffinity(1266992, 8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]) = 8
fstat(4, {st_mode=S_IFDIR|0555, st_size=0, ...}) = 0
lseek(4, 0, SEEK_SET)                   = 0
getdents64(4, 0x2bb8e110 /* 4 entries */, 32768) = 112
getdents64(4, 0x2bb8e110 /* 0 entries */, 32768) = 0
close(4)                                = 0
sched_getaffinity(0, 8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]) = 8
sched_setaffinity(0, 8, [0])            = 0
sched_setaffinity(0, 8, [1])            = 0
sched_setaffinity(0, 8, [2])            = 0
sched_setaffinity(0, 8, [3])            = 0
sched_setaffinity(0, 8, [4])            = 0
sched_setaffinity(0, 8, [5])            = 0
sched_setaffinity(0, 8, [6])            = 0
sched_setaffinity(0, 8, [7])            = 0
sched_setaffinity(0, 8, [8])            = 0
sched_setaffinity(0, 8, [9])            = 0
sched_setaffinity(0, 8, [10])           = 0
sched_setaffinity(0, 8, [11])           = 0
sched_setaffinity(0, 8, [12])           = 0
sched_setaffinity(0, 8, [13])           = 0
sched_setaffinity(0, 8, [14])           = 0
sched_setaffinity(0, 8, [15])           = 0
sched_setaffinity(0, 8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]) = 0
openat(AT_FDCWD, "/proc/self/task", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 4
fstat(4, {st_mode=S_IFDIR|0555, st_size=0, ...}) = 0
fstat(4, {st_mode=S_IFDIR|0555, st_size=0, ...}) = 0
lseek(4, 0, SEEK_SET)                   = 0
getdents64(4, 0x2bb92820 /* 4 entries */, 32768) = 112
getdents64(4, 0x2bb92820 /* 0 entries */, 32768) = 0
sched_getaffinity(1266982, 8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]) = 8
sched_getaffinity(1266992, 8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]) = 8
fstat(4, {st_mode=S_IFDIR|0555, st_size=0, ...}) = 0
lseek(4, 0, SEEK_SET)                   = 0
getdents64(4, 0x2bb92820 /* 4 entries */, 32768) = 112
getdents64(4, 0x2bb92820 /* 0 entries */, 32768) = 0
close(4)                                = 0
openat(AT_FDCWD, "/proc/self/task", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 4
fstat(4, {st_mode=S_IFDIR|0555, st_size=0, ...}) = 0
fstat(4, {st_mode=S_IFDIR|0555, st_size=0, ...}) = 0
lseek(4, 0, SEEK_SET)                   = 0
getdents64(4, 0x2bb92870 /* 4 entries */, 32768) = 112
getdents64(4, 0x2bb92870 /* 0 entries */, 32768) = 0
sched_getaffinity(1266982, 8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]) = 8
sched_getaffinity(1266992, 8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]) = 8
fstat(4, {st_mode=S_IFDIR|0555, st_size=0, ...}) = 0
lseek(4, 0, SEEK_SET)                   = 0
getdents64(4, 0x2bb92870 /* 4 entries */, 32768) = 112
getdents64(4, 0x2bb92870 /* 0 entries */, 32768) = 0
close(4)                                = 0
mmap(NULL, 8392704, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7f3a7b7ff000
mprotect(0x7f3a7b800000, 8388608, PROT_READ|PROT_WRITE) = 0
rt_sigprocmask(SIG_BLOCK, ~[], [], 8)   = 0
clone3({flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM|CLONE_SETTLS|CLONE_PARENT_SETTID|CLONE_CHILD_CLEARTID, child_tid=0x7f3a7bfff990, parent_tid=0x7f3a7bfff990, exit_signal=0, stack=0x7f3a7b7ff000, stack_size=0x7fff80, tls=0x7f3a7bfff6c0} => {parent_tid=[1267004]}, 88) = 1267004
rt_sigprocmask(SIG_SETMASK, [], NULL, 8) = 0
futex(0x2bb92980, FUTEX_WAIT, 2147483648, NULL) = 0
futex(0x2bb83780, FUTEX_WAKE_PRIVATE, 2147483647) = 1
futex(0x7f3a7bfff990, FUTEX_WAIT_BITSET|FUTEX_CLOCK_REALTIME, 1267004, NULL, FUTEX_BITSET_MATCH_ANY) = 0
readlink("/home", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_cpu_plugin.so", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_cpu_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 52356657
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_gpu_plugin.so", O_RDONLY|O_CLOEXEC) = 4
read(4, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0 \227\33\0\0\0\0\0"..., 832) = 832
fstat(4, {st_mode=S_IFREG|0644, st_size=37232649, ...}) = 0
mmap(NULL, 38166536, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7f3a79200000
mprotect(0x7f3a793b5000, 36253696, PROT_NONE) = 0
mmap(0x7f3a793b5000, 24301568, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x1b5000) = 0x7f3a793b5000
mmap(0x7f3a7aae2000, 10420224, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x18e2000) = 0x7f3a7aae2000
mmap(0x7f3a7b4d3000, 593920, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x22d2000) = 0x7f3a7b4d3000
mmap(0x7f3a7b564000, 932456, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a7b564000
mmap(0x7f3a7b648000, 126976, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x2364000) = 0x7f3a7b648000
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/glibc-hwcaps/x86-64-v3/libOpenCL.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/glibc-hwcaps/x86-64-v3/", 0x7ffeca3750b0, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/glibc-hwcaps/x86-64-v2/libOpenCL.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/glibc-hwcaps/x86-64-v2/", 0x7ffeca3750b0, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libOpenCL.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/", {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}, 0) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/libOpenCL.so.1", O_RDONLY|O_CLOEXEC) = 4
read(4, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(4, {st_mode=S_IFREG|0755, st_size=277928, ...}) = 0
mmap(NULL, 230952, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7f3a8034e000
mmap(0x7f3a80355000, 135168, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x7000) = 0x7f3a80355000
mmap(0x7f3a80376000, 61440, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x28000) = 0x7f3a80376000
mmap(0x7f3a80385000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x37000) = 0x7f3a80385000
close(4)                                = 0
mprotect(0x7f3a80385000, 4096, PROT_READ) = 0
mprotect(0x7f3a7b4d3000, 573440, PROT_READ) = 0
brk(0x2bbc6000)                         = 0x2bbc6000
brk(0x2bbe7000)                         = 0x2bbe7000
brk(0x2bc08000)                         = 0x2bc08000
brk(0x2bc2a000)                         = 0x2bc2a000
brk(0x2bc4c000)                         = 0x2bc4c000
brk(0x2bc6d000)                         = 0x2bc6d000
brk(0x2bc94000)                         = 0x2bc94000
brk(0x2bcb5000)                         = 0x2bcb5000
brk(0x2bcd6000)                         = 0x2bcd6000
brk(0x2bd01000)                         = 0x2bd01000
brk(0x2bd2b000)                         = 0x2bd2b000
brk(0x2bd54000)                         = 0x2bd54000
brk(0x2bd75000)                         = 0x2bd75000
brk(0x2bd96000)                         = 0x2bd96000
brk(0x2bdb7000)                         = 0x2bdb7000
brk(0x2bddf000)                         = 0x2bddf000
brk(0x2be00000)                         = 0x2be00000
brk(0x2be25000)                         = 0x2be25000
brk(0x2be46000)                         = 0x2be46000
brk(0x2be68000)                         = 0x2be68000
brk(0x2be8c000)                         = 0x2be8c000
brk(0x2bebc000)                         = 0x2bebc000
brk(0x2bee2000)                         = 0x2bee2000
brk(0x2bf06000)                         = 0x2bf06000
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/etc/OpenCL/vendors", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 4
fstat(4, {st_mode=S_IFDIR|S_ISGID|0755, st_size=4096, ...}) = 0
getdents64(4, 0x2befd8d0 /* 3 entries */, 32768) = 80
stat("/home/loganr/.conda/envs/fly/etc/OpenCL/vendors/.conda_keep", {st_mode=S_IFREG|0644, st_size=0, ...}) = 0
getdents64(4, 0x2befd8d0 /* 0 entries */, 32768) = 0
lseek(4, 0, SEEK_SET)                   = 0
close(4)                                = 0
readlink("/home", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_gpu_plugin.so", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_gpu_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 37232649
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_hetero_plugin.so", O_RDONLY|O_CLOEXEC) = 4
read(4, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\360\377\0\0\0\0\0\0"..., 832) = 832
fstat(4, {st_mode=S_IFREG|0644, st_size=484329, ...}) = 0
mmap(NULL, 488424, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7f3a802d6000
mmap(0x7f3a802e5000, 319488, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0xf000) = 0x7f3a802e5000
mmap(0x7f3a80333000, 69632, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x5d000) = 0x7f3a80333000
mmap(0x7f3a80344000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x6d000) = 0x7f3a80344000
mmap(0x7f3a80348000, 24576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x71000) = 0x7f3a80348000
close(4)                                = 0
mprotect(0x7f3a80344000, 12288, PROT_READ) = 0
readlink("/home", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_hetero_plugin.so", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_hetero_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 484329
close(4)                                = 0
readlink("/home", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_plugin.so", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 565745
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_npu_plugin.so", O_RDONLY|O_CLOEXEC) = 4
read(4, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\260H\6\0\0\0\0\0"..., 832) = 832
fstat(4, {st_mode=S_IFREG|0644, st_size=5884289, ...}) = 0
mmap(NULL, 5904768, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7f3a78c00000
mprotect(0x7f3a78c61000, 5382144, PROT_NONE) = 0
mmap(0x7f3a78c61000, 4366336, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x61000) = 0x7f3a78c61000
mmap(0x7f3a7908b000, 913408, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x48b000) = 0x7f3a7908b000
mmap(0x7f3a7916a000, 86016, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x569000) = 0x7f3a7916a000
mmap(0x7f3a7917f000, 13640, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f3a7917f000
mmap(0x7f3a79183000, 126976, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x57e000) = 0x7f3a79183000
close(4)                                = 0
mprotect(0x7f3a7916a000, 81920, PROT_READ) = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libze_loader.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/libze_loader.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 4
fstat(4, {st_mode=S_IFREG|0644, st_size=243639, ...}) = 0
mmap(NULL, 243639, PROT_READ, MAP_PRIVATE, 4, 0) = 0x7f3a8029a000
close(4)                                = 0
openat(AT_FDCWD, "/usr/lib/libze_loader.so.1", O_RDONLY|O_CLOEXEC) = 4
read(4, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
fstat(4, {st_mode=S_IFREG|0755, st_size=780408, ...}) = 0
mmap(NULL, 778536, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7f3a801db000
mmap(0x7f3a801ee000, 569344, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x13000) = 0x7f3a801ee000
mmap(0x7f3a80279000, 122880, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x9e000) = 0x7f3a80279000
mmap(0x7f3a80297000, 12288, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0xbc000) = 0x7f3a80297000
close(4)                                = 0
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/bin/../lib/libspdlog.so.1.17", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/usr/lib/libspdlog.so.1.17", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
munmap(0x7f3a8029a000, 243639)          = 0
munmap(0x7f3a801db000, 778536)          = 0
brk(0x2bf27000)                         = 0x2bf27000
readlink("/home", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
readlink("/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_npu_plugin.so", 0x7ffeca375c20, 1023) = -1 EINVAL (Invalid argument)
openat(AT_FDCWD, "/home/loganr/.conda/envs/fly/lib/python3.10/site-packages/openvino/libs/libopenvino_intel_npu_plugin.so", O_RDONLY) = 4
lseek(4, 0, SEEK_END)                   = 5884289
close(4)                                = 0
munmap(0x7f3a802d6000, 488424)          = 0
sigaltstack(NULL, {ss_sp=0x7f3a88b19000, ss_flags=0, ss_size=11824}) = 0
sigaltstack({ss_sp=NULL, ss_flags=SS_DISABLE, ss_size=0}, NULL) = 0
munmap(0x7f3a88b19000, 11824)           = 0
munmap(0x7f3a83a73000, 569840)          = 0
write(1, "['CPU']\n", 8)                = 8
futex(0x7f3a7c000be0, FUTEX_WAIT_BITSET_PRIVATE|FUTEX_CLOCK_REALTIME, 0, NULL, FUTEX_BITSET_MATCH_ANY) = 0
rt_sigaction(SIGINT, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=SA_RESTORER|SA_ONSTACK, sa_restorer=0x7f3a8943e4d0}, {sa_handler=0x495e37, sa_mask=[], sa_flags=SA_RESTORER|SA_ONSTACK, sa_restorer=0x7f3a8943e4d0}, 8) = 0
munmap(0x7f3a8616f000, 593920)          = 0
munmap(0x7f3a8491b000, 35456)           = 0
munmap(0x7f3a8078b000, 475752)          = 0
munmap(0x7f3a8068a000, 1052672)         = 0
exit_group(0)                           = ?
+++ exited with 0 +++
