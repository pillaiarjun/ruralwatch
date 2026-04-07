# --- src/utils/spark_session.py ---
import os
from pyspark.sql import SparkSession

def get_spark():
    '''Creates a Spark session configured for Delta Lake.

    Call this at the top of every notebook that uses PySpark.
    '''
    os.environ['JAVA_HOME'] = '/opt/homebrew/opt/openjdk@17'
    # ↑ Tells Python where Java is installed.
    # On Apple Silicon with Homebrew, openjdk@17 lives here.
    # Without this, PySpark can't find Java and throws a JAVA_HOME error.

    spark = (
        SparkSession.builder
        .appName('RuralWatch')
        # ↑ Names this Spark session. Appears in logs and the Spark UI.

        .config('spark.jars.packages', 'io.delta:delta-spark_2.12:3.1.0')
        # ↑ Tells Spark to download the Delta Lake JAR from Maven on first run.
        # The format is groupId:artifactId:version.
        # 2.12 is the Scala version Spark 3.5.1 was compiled against.

        .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension')
        # ↑ Registers Delta's SQL extensions so Spark understands Delta-specific
        # SQL syntax like DESCRIBE HISTORY and RESTORE.

        .config('spark.sql.catalog.spark_catalog',
                'org.apache.spark.sql.delta.catalog.DeltaCatalog')
        # ↑ Replaces Spark's default catalog with Delta's catalog.
        # This is what lets you write .format('delta') instead of .format('parquet').

        .config('spark.driver.extraJavaOptions', '-Dlog4j.rootCategory=WARN,console')
        # ↑ Suppresses the wall of INFO logs Spark prints by default.
        # Keeps your notebook output readable.

        .getOrCreate()
        # ↑ Creates a new session, or returns the existing one if already running.
    )

    spark.sparkContext.setLogLevel('WARN')
    # ↑ Second layer of log suppression — this one applies to the SparkContext
    # level, complementing the log4j config above.

    return spark