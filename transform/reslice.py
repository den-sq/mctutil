
import click
import numpy as np
import tifffile as tf


Coord = namedtuple("Coord", ["x", "y", "z"])

# Click Parameter:
class Coordinates(click.ParamType):
	name = "Integer Coodrinates"

	def convert(self, value, param, ctx):
		try:
			coord = Coord(*[int(x) for x in value.split(",")])
			return coord
		except (ValueError, TypeError):
			self.fail(f'{value} is not a 3-value intteger coordinate.')


COORDINATES = Coordinates()



def byteread_helper(target: SharedNP, image: PathLike, i_dtype: np.dtype, offsets: ArrayLike, size: int):
	""" Sinogram order-capable reader using direct buffer reading.

		TODO: Figure out if you can get rid of for loop - order matters.

		:param target: Shared memory to read into.
		:param image: Path to image to read from.
		:param i_dtype: Data type of image.  Unused here, since reading is direct buffer.
		:param offsets: Array of memory offsets to read into.  Should be sequential in file.
		:param size: Size of chunk of memory to read.
	"""
	sm = shared_memory.SharedMemory(name=target)
	with open(image, "rb") as handle:
		handle.seek(offsets[0]["source"])
		for offset in offsets:
			handle.readinto(sm.buf[offset["target"]:offset["target"] + size])
	sm.close()


def distribute_read(target_mem: SharedNP, pj: Mapping, window, int_window,
					image_order: ArrayLike, thread_max: int = cpu_count(),
					read_func: Callable = byteread_helper, sino_order: bool = True):
	""" Distributes file reading across multiple threads.

		Currently can work in projection or sinogram order, maybe?

		:param target_mem: Memory to read files into.
		:param mem_shape: Shape of memory we are reading into.
		:param pj: Information about tiff file structure grabbed from first.
		:param window: Vertical portion of images to fetch.
		:param int_window: Internal memory range matching window.
		:param int_offset: Internal memory offset before start of internal window.
		:param image_order: Order of images to read; may not be directly followed due to starmapping.
		:param thread_max: Maximum # of threads to use.
	"""
	# Steps for memory space jumps.
	h_step = pj["x"] * pj["bytesize"]

	# Size of Sinogram Block
	sino_block_size = target_mem.shape.Theta * h_step

	# Size of Proj block
	proj_block_size = len(int_window) * h_step

	# Find the offset values for start of blocks.
	# This is hilariouslyy stupid and needs a rewrite
	base_offset = target_mem[int_window].buffer_address.start

	def generate_offset_pairs_sino(i):
		return [{"source": pj["offset"] + (window.start + j) * h_step,
				"target": int(base_offset + j * sino_block_size + i * h_step)}
					for j in range(len(int_window))]

	def generate_offset_pairs_proj(i):
		return [{"source": pj["offset"] + (window.start) * h_step, "target": int(base_offset + i * proj_block_size)}]

	if sino_order:
		log.log("Files Into Memory", f"Writing (in {target_mem.name} | {target_mem.shape}) {base_offset}"
			+ f" to {base_offset + len(int_window) * sino_block_size}", log_level=log.DEBUG.INFO)
		pairs_func = generate_offset_pairs_sino
		size = h_step
	else:
		log.log("Files Into Memory", f"Writing (in {target_mem.name} | {target_mem.shape}) {base_offset}"
			+ f" to {base_offset + len(int_window) * proj_block_size} out of {target_mem[int_window].buffer_address}",
			log_level=log.DEBUG.INFO)
		pairs_func = generate_offset_pairs_proj
		size = proj_block_size

	# Load initial data.
	with Pool(thread_max) as pool:
		pool.starmap(read_func,


@click.command()
@click.option("--reslice", "-r", type=COORDINATES, required=True,
				help="")
@click.argument("input_folder", type=click.Path(exists=True, path_type=Path))
@click.argument("output_folder", type=click.Path(path_type=Path))
def reslice()
	output_folder.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
	reslice()