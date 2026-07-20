package org.janelia.saalfeldlab.n5.precomputed;

import java.awt.GraphicsEnvironment;
import java.util.Arrays;

import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;

import ij.ImageJ;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.real.FloatType;

/**
 * Real-life example: open a Neuroglancer precomputed <em>deformation / warp
 * field</em> from Google Cloud Storage and display it in ImageJ.
 * <p>
 * The volume at {@code gs://janelia-spark-test/warp_flow_precomputed/w61_s109_r00}
 * is {@code float32}, 2 channels (the x/y displacement components), single scale,
 * raw-encoded and sharded with {@code data_encoding: gzip} — so this exercises the
 * float32 + sharded + gzip-data read path over a cloud backend. The reader presents
 * it as a 4D {@code [x, y, z, channel]} dataset, which ImageJ shows as a hyperstack.
 * <p>
 * Run interactively:
 * <pre>
 * mvn test-compile exec:java -Dexec.classpathScope=test \
 *   -Dexec.mainClass=org.janelia.saalfeldlab.n5.precomputed.WarpFieldPrecomputed
 * </pre>
 *
 * @author Stephan Preibisch
 */
public class WarpFieldPrecomputed {

	public static final String URL =
			"precomputed://gs://janelia-spark-test/warp_flow_precomputed/w61_s109_r00";

	public static void main(final String[] args) {

		final String url = args.length > 0 ? args[0] : URL;
		final N5Reader n5 = HemibrainPrecomputed.open(url);

		final String key = n5.list("/")[0];
		final DatasetAttributes attrs = n5.getDatasetAttributes(key);
		System.out.println("scale " + key + ": " + attrs.getDataType()
				+ " " + Arrays.toString(attrs.getDimensions())
				+ " block " + Arrays.toString(attrs.getBlockSize()));

		final RandomAccessibleInterval<FloatType> field = N5Utils.open(n5, key);

		// probe the center voxel to force a real (sharded, gzip, float32) chunk read
		final long[] dims = field.dimensionsAsLongArray();
		final RandomAccess<FloatType> ra = field.randomAccess();
		ra.setPosition(new long[]{dims[0] / 2, dims[1] / 2, dims[2] / 2, 0});
		System.out.println("center sample (channel 0) = " + ra.get().get());

		if (GraphicsEnvironment.isHeadless()) {
			System.out.println("headless environment: skipping ImageJ display");
			return;
		}

		new ImageJ();
		ImageJFunctions.show(field, "w61_s109_r00");
	}
}
