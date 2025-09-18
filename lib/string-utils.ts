/**
 * Safely truncates a filename while preserving the extension
 * @param filename The filename to truncate
 * @param maxLength The maximum length of the resulting filename (default: 24)
 * @returns The truncated filename with extension preserved
 */
export function truncateFilename(filename: string, maxLength: number = 24): string {
  if (filename.length <= maxLength) return filename;
  
  // Find the last dot to separate name and extension
  const lastDotIndex = filename.lastIndexOf('.');
  
  // Handle files without extension or files starting with a dot (like .gitignore)
  if (lastDotIndex === -1 || lastDotIndex === 0) {
    // No extension or hidden file, just truncate the name
    return filename.substring(0, maxLength - 3) + '...';
  }
  
  const name = filename.slice(0, lastDotIndex);
  const ext = filename.slice(lastDotIndex + 1);
  const extLen = ext.length;
  
  // Calculate available space for the name part
  // We need space for: name + '...' + '.' + extension
  const availableForName = maxLength - extLen - 4; // 4 = '...' + '.'
  
  // If the available space is too small, just truncate without extension
  if (availableForName < 1) {
    return filename.substring(0, maxLength - 3) + '...';
  }
  
  const truncatedName = name.substring(0, availableForName) + '...';
  return `${truncatedName}.${ext}`;
}
