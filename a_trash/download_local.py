import pyarrow as pa
import pyarrow.parquet as pq
import os

def inspect_arrow_files():
    """Inspect the Arrow files directly"""
    
    arrow_dir = "./katanaml-org___invoices-donut-data-v1/default/0.0.0/d2cde298e79c94fb05bc320999deb4b7889b0464"
    
    for file in os.listdir(arrow_dir):
        if file.endswith('.arrow'):
            file_path = os.path.join(arrow_dir, file)
            print(f"\n📄 File: {file}")
            
            try:
                # Try to read as Arrow file
                reader = pa.ipc.open_file(file_path)
                print(f"   Schema: {reader.schema}")
                print(f"   Num records: {reader.num_record_batches}")
            except Exception as e:
                print(f"   Error reading: {e}")

# inspect_arrow_files()