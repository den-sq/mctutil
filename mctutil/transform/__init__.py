"""Unified transform command group."""

from mctutil.lazy import LazyGroup

main = LazyGroup(
	name="transform",
	help="TIFF stack transforms and related local data reshaping helpers.",
	lazy_subcommands={
		"channelize": "mctutil.transform.channelize:channelize",
		"convert": "mctutil.transform.convert:convert",
		"df-write-tiff": "mctutil.transform.df_write_tiff:df_write_tiff",
		"dicom-conv": "mctutil.transform.dicom_conv:dicom_conv",
		"downsample": "mctutil.transform.downsample:downsample",
		"find-bounds": "mctutil.transform.find_bounds:find_bounds",
		"fix-name": "mctutil.transform.fix_name:fix_names",
		"flip": "mctutil.transform.flip:flip_stack",
		"hdf-convert": "mctutil.transform.hdf_convert:hdf_convert",
		"normalize": "mctutil.transform.normalize:norm",
		"reslice": "mctutil.transform.reslice:reslice",
		"denoise": "mctutil.transform.simple_noise:simple_denoise",
		"stitch": "mctutil.transform.stitch:stitch",
		"transpose": "mctutil.transform.transpose:transpose_stack",
		"trim": "mctutil.transform.trim:trim",
		"decompress-tiff": "mctutil.transform.uncompress:uncompress",
		"gunzip": "mctutil.transform.quickgunzip:gunzip",
		"strip-gz-suffix": "mctutil.transform.gz_strip:stripgz",
	},
)
