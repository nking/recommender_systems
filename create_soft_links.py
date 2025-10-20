import os

cwd = os.getcwd()

bin_path = os.path.join(cwd, "bin/")

os.makedirs(bin_path, exist_ok=True)

os.chdir(bin_path)

print(f'{os.getcwd()}')

src_dir = os.path.join(bin_path, "../src/main/python/movie_lens_tfx/")
test_dir = os.path.join(bin_path, "../src/test/python/movie_lens_tfx/")

for dir_path in [src_dir, test_dir]:
  for file_name in os.listdir(dir_path):
    file_path = os.path.join(dir_path, file_name)
    if os.path.isfile(file_path) and file_name != "__init__.py"\
      and file_name.endswith(".py"):
      try:
        os.symlink(file_path, file_name)
      except Exception as e:
        print(f"ignore if expected: {e}")
    
