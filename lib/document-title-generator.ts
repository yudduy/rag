/**
 * Document Title Generator
 * Generates semantic titles for documents based on their content
 */

export interface DocumentTitleOptions {
  maxLength?: number;
  fallbackToFilename?: boolean;
}

export class DocumentTitleGenerator {
  /**
   * Generate a semantic title from document content
   */
  static generateTitle(
    content: string, 
    filename: string, 
    fileType: string, 
    options: DocumentTitleOptions = {}
  ): string {
    const { maxLength = 60, fallbackToFilename = true } = options;
    
    try {
      // Clean and normalize content for analysis
      const cleanContent = content.trim().replace(/\s+/g, ' ').substring(0, 2000);
      
      if (!cleanContent) {
        return fallbackToFilename ? this.cleanFilename(filename) : 'Untitled Document';
      }

      // Try different strategies based on file type and content
      let title = '';

      // Strategy 1: Look for explicit titles
      title = this.extractExplicitTitle(cleanContent, fileType);
      if (title) return this.truncateTitle(title, maxLength);

      // Strategy 2: Resume/CV detection
      title = this.extractResumeTitle(cleanContent);
      if (title) return this.truncateTitle(title, maxLength);

      // Strategy 3: Research paper detection
      title = this.extractPaperTitle(cleanContent);
      if (title) return this.truncateTitle(title, maxLength);

      // Strategy 4: Extract from first meaningful sentence
      title = this.extractFromFirstSentence(cleanContent);
      if (title) return this.truncateTitle(title, maxLength);

      // Strategy 5: Extract key phrases
      title = this.extractKeyPhrases(cleanContent);
      if (title) return this.truncateTitle(title, maxLength);

      // Fallback to cleaned filename
      return fallbackToFilename ? this.cleanFilename(filename) : 'Untitled Document';
      
    } catch (error) {
      console.error('Error generating document title:', error);
      return fallbackToFilename ? this.cleanFilename(filename) : 'Untitled Document';
    }
  }

  /**
   * Extract explicit titles from common document patterns
   */
  private static extractExplicitTitle(content: string, fileType: string): string {
    const patterns = [
      // Markdown titles
      /^#\s+(.+?)$/m,
      /^Title:\s*(.+?)$/mi,
      /^Subject:\s*(.+?)$/mi,
      
      // Common document headers
      /^\s*(.+?)\s*\n\s*={3,}/m,
      /^\s*(.+?)\s*\n\s*-{3,}/m,
      
      // PDF extracted titles (often appear at the beginning)
      /^(.+?)\s*\n/m,
    ];

    for (const pattern of patterns) {
      const match = content.match(pattern);
      if (match && match[1]) {
        const title = match[1].trim();
        if (this.isValidTitle(title)) {
          return title;
        }
      }
    }

    return '';
  }

  /**
   * Extract resume/CV titles
   */
  private static extractResumeTitle(content: string): string {
    const lowerContent = content.toLowerCase();
    
    // Check if this looks like a resume
    const resumeIndicators = [
      'resume', 'curriculum vitae', 'cv', 'experience', 'education', 
      'skills', 'employment', 'work history', 'professional'
    ];
    
    const hasResumeIndicators = resumeIndicators.some(indicator => 
      lowerContent.includes(indicator)
    );

    if (!hasResumeIndicators) return '';

    // Look for name patterns at the beginning
    const lines = content.split('\n').slice(0, 10);
    
    for (const line of lines) {
      const cleanLine = line.trim();
      if (cleanLine.length < 5 || cleanLine.length > 50) continue;
      
      // Skip common resume headers
      if (/^(resume|curriculum vitae|cv)$/i.test(cleanLine)) continue;
      
      // Look for name patterns
      if (this.looksLikeName(cleanLine)) {
        return `${cleanLine} - Resume`;
      }
      
      // Look for "Name's Resume" patterns
      const nameMatch = cleanLine.match(/^(.+?)'?s?\s+(resume|cv)$/i);
      if (nameMatch) {
        return `${nameMatch[1].trim()} - Resume`;
      }
    }

    return 'Professional Resume';
  }

  /**
   * Extract research paper titles
   */
  private static extractPaperTitle(content: string): string {
    const lowerContent = content.toLowerCase();
    
    // Check if this looks like a research paper
    const paperIndicators = [
      'abstract', 'introduction', 'methodology', 'results', 'conclusion',
      'references', 'bibliography', 'doi:', 'arxiv:', 'journal'
    ];
    
    const hasPaperIndicators = paperIndicators.some(indicator => 
      lowerContent.includes(indicator)
    );

    if (!hasPaperIndicators) return '';

    // Look for title before abstract
    const abstractMatch = content.match(/^(.*?)\s*abstract\s*:?\s*/is);
    if (abstractMatch) {
      const beforeAbstract = abstractMatch[1].trim();
      const lines = beforeAbstract.split('\n').filter(line => line.trim());
      
      // Usually the title is one of the first few substantial lines
      for (const line of lines.slice(0, 5)) {
        const cleanLine = line.trim();
        if (cleanLine.length > 10 && cleanLine.length < 200 && this.isValidTitle(cleanLine)) {
          return cleanLine;
        }
      }
    }

    return '';
  }

  /**
   * Extract title from first meaningful sentence
   */
  private static extractFromFirstSentence(content: string): string {
    const sentences = content.split(/[.!?]+/).map(s => s.trim()).filter(s => s.length > 5);
    
    for (const sentence of sentences.slice(0, 3)) {
      if (sentence.length > 10 && sentence.length < 100 && this.isValidTitle(sentence)) {
        return sentence;
      }
    }

    return '';
  }

  /**
   * Extract key phrases using simple NLP
   */
  private static extractKeyPhrases(content: string): string {
    const words = content.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 3);

    // Count word frequency
    const wordCount = new Map<string, number>();
    words.forEach(word => {
      wordCount.set(word, (wordCount.get(word) || 0) + 1);
    });

    // Get top words (excluding common words)
    const commonWords = new Set([
      'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'
    ]);

    const topWords = Array.from(wordCount.entries())
      .filter(([word]) => !commonWords.has(word))
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([word]) => word);

    if (topWords.length >= 2) {
      return topWords.map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    }

    return '';
  }

  /**
   * Check if a string looks like a person's name
   */
  private static looksLikeName(text: string): boolean {
    // Simple heuristics for name detection
    const words = text.trim().split(/\s+/);
    
    // Should be 2-4 words
    if (words.length < 2 || words.length > 4) return false;
    
    // Each word should be capitalized and contain only letters
    return words.every(word => 
      /^[A-Z][a-z]+$/.test(word) && word.length > 1
    );
  }

  /**
   * Check if a string is a valid title
   */
  private static isValidTitle(title: string): boolean {
    if (!title || title.length < 3 || title.length > 200) return false;
    
    // Avoid titles that are just numbers, dates, or single words
    if (/^\d+$/.test(title)) return false;
    if (/^\d{1,2}\/\d{1,2}\/\d{2,4}$/.test(title)) return false;
    if (!/\s/.test(title) && title.length < 8) return false;
    
    // Avoid common non-title patterns
    const badPatterns = [
      /^page\s+\d+/i,
      /^chapter\s+\d+/i,
      /^section\s+\d+/i,
      /^figure\s+\d+/i,
      /^table\s+\d+/i,
    ];
    
    return !badPatterns.some(pattern => pattern.test(title));
  }

  /**
   * Clean filename for display
   */
  private static cleanFilename(filename: string): string {
    return filename
      .replace(/\.[^.]+$/, '') // Remove extension
      .replace(/[_-]/g, ' ') // Replace underscores and hyphens with spaces
      .replace(/([a-z])([A-Z])/g, '$1 $2') // Add spaces before capital letters
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ')
      .trim();
  }

  /**
   * Truncate title to specified length
   */
  private static truncateTitle(title: string, maxLength: number): string {
    if (title.length <= maxLength) return title;
    
    const truncated = title.substring(0, maxLength - 3);
    const lastSpace = truncated.lastIndexOf(' ');
    
    if (lastSpace > maxLength * 0.7) {
      return truncated.substring(0, lastSpace) + '...';
    }
    
    return truncated + '...';
  }
}
