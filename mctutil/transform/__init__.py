"""Unified transform command group."""

from mctutil.lazy import LazyGroup

main = LazyGroup(
	name="transform",
	help="TIFF stack transforms and related local data reshaping helpers.",
	lazy_subcommands={
		"channelize": "transform.channelize:channelize",
		"convert": "transform.convert:convert",
		"df-write-tiff": "transform.df_write_tiff:df_write_tiff",
		"dicom-conv": "transform.dicom_conv:dicom_conv",
		"downsample": "transform.downsample:downsample",
		"find-bounds": "transform.find_bounds:find_bounds",
		"fix-name": "transform.fix_name:fix_names",
		"hdf-convert": "transform.hdf_convert:hdf_convert",
		"normalize": "transform.normalize:norm",
		"denoise": "transform.simple_noise:simple_denoise",
		"stitch": "transform.stitch:stitch",
		"transpose": "transform.transpose:transpose_stack",
		"trim": "transform.trim:trim",
		"decompress-tiff": "transform.uncompress:uncompress",
		"gunzip": "transform.quickgunzip:gunzip",
		"strip-gz-suffix": "transform.gz_strip:stripgz",
	},
)
