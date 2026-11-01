import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# IN SILICO PCR ANALYZER APP
# This app identifies potential PCR amplicons by finding 
# Forward and Reverse primer blast hits on the same contig,
# verifying their orientation, and calculating the distance.
# ---------------------------------------------------------

st.set_page_config(page_title="In Silico PCR Analyzer", layout="wide")

st.title("🧬 In Silico PCR Analyzer")
st.markdown("""
This application takes two BLAST tabular output files (`-outfmt 6` or similar) for **Forward** and **Reverse** primers.
It merges the hits on the same contig, checks for correct strand orientations, calculates the expected **PCR product (amplicon) sizes**, and displays the locations.
""")

# Sidebar inputs for parameters
st.sidebar.header("PCR Parameters")
min_size = st.sidebar.number_input("Minimum Amplicon Size (bp)", value=100, step=50)
max_size = st.sidebar.number_input("Maximum Amplicon Size (bp)", value=3000, step=100)

# Function to load and process BLAST tabular output
def load_blast_file(uploaded_file):
    """
    Reads a BLAST outfmt 6 file and determines the mapping coordinates and strand orientation.
    """
    # Standard BLAST tabular columns
    cols = [
        "query", "subject", "identity", "align_length", "mismatches", 
        "gap_opens", "q_start", "q_end", "s_start", "s_end", "evalue", "bit_score"
    ]
    
    # Read the file using pandas
    df = pd.read_csv(uploaded_file, sep="\t", comment="#", names=cols)
    
    # Determine strand orientation based on subject start and end coordinates
    # If s_start < s_end, it's on the positive (+) strand. Otherwise, negative (-).
    df['strand'] = df.apply(lambda row: '+' if row['s_start'] < row['s_end'] else '-', axis=1)
    
    # Get the absolute min and max positions on the reference contig
    df['min_pos'] = df[['s_start', 's_end']].min(axis=1)
    df['max_pos'] = df[['s_start', 's_end']].max(axis=1)
    
    return df

def calculate_amplicon_size(row):
    """
    Calculates the expected PCR product size based on F and R primer positions.
    Valid amplicons occur when primers face each other on opposite strands.
    """
    if row['strand_F'] == '+':
        # Forward primer is on +, Reverse primer is on -
        # Amplicon spans from the start of F to the end of R
        if row['min_pos_F'] < row['max_pos_R']:
            return int(row['max_pos_R'] - row['min_pos_F'] + 1)
    elif row['strand_F'] == '-':
        # Forward primer is on -, Reverse primer is on +
        # Amplicon spans from the start of R to the end of F
        if row['min_pos_R'] < row['max_pos_F']:
            return int(row['max_pos_F'] - row['min_pos_R'] + 1)
    
    # Return -1 if they are facing away from each other (invalid PCR product)
    return -1

# File uploaders
col1, col2 = st.columns(2)
with col1:
    f_file = st.file_uploader("Upload Forward Primer BLAST result (TXT)", type=["txt", "tsv"])
with col2:
    r_file = st.file_uploader("Upload Reverse Primer BLAST result (TXT)", type=["txt", "tsv"])

if f_file and r_file:
    st.info("Files uploaded successfully. Processing data...")
    
    # Process files
    df_f = load_blast_file(f_file)
    df_r = load_blast_file(r_file)
    
    # Merge on the 'subject' (contig/chromosome) column
    merged_df = pd.merge(df_f, df_r, on="subject", suffixes=("_F", "_R"))
    
    # Filter out pairs that are on the same strand (a valid PCR needs opposite strands)
    valid_pairs = merged_df[merged_df['strand_F'] != merged_df['strand_R']].copy()
    
    if not valid_pairs.empty:
        # Calculate amplicon sizes
        valid_pairs['Amplicon_Size_(bp)'] = valid_pairs.apply(calculate_amplicon_size, axis=1)
        
        # Filter based on user-defined size limits
        final_results = valid_pairs[
            (valid_pairs['Amplicon_Size_(bp)'] >= min_size) & 
            (valid_pairs['Amplicon_Size_(bp)'] <= max_size)
        ].copy()
        
        if not final_results.empty:
            # Sort by Amplicon Size
            final_results = final_results.sort_values(by="Amplicon_Size_(bp)")
            
            # Select relevant columns for display
            display_cols = [
                "subject", 
                "Amplicon_Size_(bp)", 
                "s_start_F", "s_end_F", "strand_F", 
                "s_start_R", "s_end_R", "strand_R",
                "identity_F", "identity_R"
            ]
            
            # Rename columns for better readability
            display_df = final_results[display_cols].rename(columns={
                "subject": "Contig / Reference",
                "s_start_F": "F_Start",
                "s_end_F": "F_End",
                "strand_F": "F_Strand",
                "s_start_R": "R_Start",
                "s_end_R": "R_End",
                "strand_R": "R_Strand",
                "identity_F": "F_Identity(%)",
                "identity_R": "R_Identity(%)"
            })
            
            st.success(f"Found {len(display_df)} potential amplicon(s) matching the criteria!")
            
            # Display interactive table
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='epcr_results.csv',
                mime='text/csv',
            )
        else:
            st.warning("No amplicons found within the specified size range.")
    else:
        st.warning("No valid primer pairs (opposite strands) found on the same contigs.")