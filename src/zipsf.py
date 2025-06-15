import requests
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import string

# ===============================
# PART 2: Zipf's Law Analysis
# ===============================

def download_pride_and_prejudice():
    """Download Pride and Prejudice from Project Gutenberg"""
    
    url = "https://www.gutenberg.org/files/1342/1342-0.txt"
    
    print("Downloading 'Pride and Prejudice' from Project Gutenberg...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
        print("Download successful!")
        return text
    except requests.RequestException as e:
        print(f"Download failed: {e}")
        return None

def preprocess_text(text):
    """Clean and preprocess the text"""
    
    # Find the start and end of the actual book content
    # Remove Project Gutenberg header and footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        # Extract only the book content
        text = text[start_idx:end_idx]
        print("Removed Project Gutenberg header and footer")
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Remove empty strings
    words = [word for word in words if word.strip()]
    
    print(f"Preprocessed text: {len(words)} words")
    return words

def calculate_word_frequencies(words):
    """Calculate word frequencies and return sorted list"""
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Sort by frequency (descending)
    sorted_words = word_counts.most_common()
    
    print(f"Unique words: {len(sorted_words)}")
    print(f"Total words: {sum(word_counts.values())}")
    
    return sorted_words

def calculate_zipf_constant(word_frequencies):
    """Calculate Zipf's law constant using top 50 terms"""
    
    # Extract frequencies and ranks for top 50 terms
    top_50 = word_frequencies[:50]
    
    constants = []
    
    print("\nTop 10 words and their Zipf constants:")
    print("Rank\tWord\t\tFrequency\tZipf Constant (cf_i * i)")
    print("-" * 65)
    
    for i, (word, freq) in enumerate(top_50, 1):
        zipf_constant = freq * i  # cf_i * i = const
        constants.append(zipf_constant)
        
        if i <= 10:  # Print first 10 for verification
            print(f"{i}\t{word:12}\t{freq}\t\t{zipf_constant:.2f}")
    
    # Calculate average constant
    avg_constant = np.mean(constants)
    std_constant = np.std(constants)
    
    print(f"\nZipf Constant Statistics (top 50 terms):")
    print(f"Average constant: {avg_constant:.2f}")
    print(f"Standard deviation: {std_constant:.2f}")
    print(f"Coefficient of variation: {(std_constant/avg_constant)*100:.1f}%")
    
    return avg_constant, constants

def plot_zipf_distribution(word_frequencies):
    """Plot term frequency vs rank for top 50 terms"""
    
    # Extract top 50 terms and frequencies
    top_50 = word_frequencies[:50]
    terms = [word for word, freq in top_50]
    frequencies = [freq for word, freq in top_50]
    ranks = list(range(1, 51))
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Bar chart of frequencies
    plt.subplot(1, 2, 1)
    plt.bar(ranks, frequencies, color='steelblue', alpha=0.7)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Word Frequency by Rank (Top 50 Terms)')
    plt.grid(True, alpha=0.3)
    
    # Add some term labels for top 10
    for i in range(10):
        plt.annotate(terms[i], (ranks[i], frequencies[i]), 
                    rotation=45, ha='center', va='bottom', fontsize=8)
    
    # Subplot 2: Log-log plot to visualize Zipf's law
    plt.subplot(1, 2, 2)
    plt.loglog(ranks, frequencies, 'bo-', markersize=4, linewidth=1, label='Observed')
    
    # Add theoretical Zipf line (using calculated constant)
    const = frequencies[0] * ranks[0]  # Use first term to estimate constant
    theoretical_freq = [const / r for r in ranks]
    plt.loglog(ranks, theoretical_freq, 'r--', linewidth=2, label=f'Zipf Law (const={const:.0f})')
    
    plt.xlabel('Rank (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Zipf\'s Law: Log-Log Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('zipf_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return terms, frequencies

def analyze_zipf_fit(word_frequencies):
    """Analyze how well the data fits Zipf's law"""
    
    top_50 = word_frequencies[:50]
    ranks = np.array(range(1, 51))
    frequencies = np.array([freq for word, freq in top_50])
    
    # Calculate theoretical frequencies using Zipf's law
    # Using the most frequent word to set the constant
    const = frequencies[0] * ranks[0]
    theoretical_freq = const / ranks
    
    # Calculate R-squared
    ss_res = np.sum((frequencies - theoretical_freq) ** 2)
    ss_tot = np.sum((frequencies - np.mean(frequencies)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate mean absolute percentage error
    mape = np.mean(np.abs((frequencies - theoretical_freq) / frequencies)) * 100
    
    print(f"\nZipf's Law Fit Analysis:")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    if r_squared > 0.8:
        print("✓ Good fit to Zipf's law")
    elif r_squared > 0.6:
        print("~ Moderate fit to Zipf's law")
    else:
        print("✗ Poor fit to Zipf's law")
    
    return r_squared, mape

def run_zipf_analysis():
    """Main function to run complete Zipf analysis"""
    
    print("=" * 60)
    print("INFORMATION RETRIEVAL PROJECT - PART 2")
    print("Zipf's Law Analysis on 'Pride and Prejudice'")
    print("=" * 60)
    
    # Step 1: Download the text
    text = download_pride_and_prejudice()
    if text is None:
        print("Failed to download text. Exiting.")
        return
    
    print(f"\nOriginal text length: {len(text)} characters")
    
    # Step 2: Preprocess the text
    words = preprocess_text(text)
    
    # Step 3: Calculate word frequencies
    word_frequencies = calculate_word_frequencies(words)
    
    # Step 4: Calculate Zipf constant
    zipf_constant, constants = calculate_zipf_constant(word_frequencies)
    
    # Step 5: Plot the results
    terms, frequencies = plot_zipf_distribution(word_frequencies)
    
    # Step 6: Analyze fit quality
    r_squared, mape = analyze_zipf_fit(word_frequencies)
    
    # Step 7: Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY:")
    print(f"• Zipf Constant: {zipf_constant:.2f}")
    print(f"• Most frequent word: '{word_frequencies[0][0]}' (frequency: {word_frequencies[0][1]})")
    print(f"• Zipf's law fit: R² = {r_squared:.4f}")
    print(f"• Top 50 terms plotted successfully")
    print("=" * 60)
    
    return word_frequencies, zipf_constant, terms, frequencies

if __name__ == "__main__":
    # Run the complete Zipf analysis
    results = run_zipf_analysis()