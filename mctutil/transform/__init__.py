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
		"downsample": "mctutil.transform.convert:downsample",
		"find-bounds": "mctutil.transform.find_bounds:find_bounds",
		"fix-name": "mctutil.transform.fix_name:fix_names",
		"flip": "mctutil.transform.flip:flip_stack",
		"h5-convert": "mctutil.transform.h5_convert:h5_convert",
		"hdf-convert": "mctutil.transform.hdf_convert:hdf_convert",
		"memmap-prep": "mctutil.transform.memmap_prep:memmap_prep",
		"normalize": "mctutil.transform.normalize:norm",
		"pipeline": "mctutil.transform.pipeline:pipeline",
		"raw-convert": "mctutil.transform.raw_convert:raw_convert",
		"reslice": "mctutil.transform.reslice:reslice",
		"denoise": "mctutil.transform.simple_noise:simple_denoise",
		"stack-split": "mctutil.transform.stack_split:stack_split",
		"stitch": "mctutil.transform.stitch:stitch",
		"stitch-reconstructions": (
			"mctutil.transform.stitch_reconstructions:stitch_reconstructions"
		),
		"transpose": "mctutil.transform.transpose:transpose_stack",
		"trim": "mctutil.transform.trim:trim",
		"decompress-tiff": "mctutil.transform.uncompress:uncompress",
		"gunzip": "mctutil.transform.quickgunzip:gunzip",
		"strip-gz-suffix": "mctutil.transform.gz_strip:stripgz",
	},
)
