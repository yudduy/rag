"""Query analysis and routing for complex RAG workflows.

Handles query classification, decomposition of complex questions into
sub-queries, and intelligent routing to appropriate processing strategies.
"""

import os
import re
import asyncio
import logging
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from llama_index.core.llms.llm import LLM
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Different types of user queries we can identify."""
    FACTUAL = "factual"          # Simple facts: "What is X?"
    SEMANTIC = "semantic"        # Meaning-based queries
    COMPARATIVE = "comparative"  # "Compare X and Y"
    PROCEDURAL = "procedural"    # "How do I...?"
    ANALYTICAL = "analytical"    # "Why does X happen?"
    MULTIFACETED = "multifaceted"  # Multiple questions in one


@dataclass
class QueryClassification:
    """What we learned about a user's query."""
    query_type: QueryType
    confidence: float
    complexity_score: float
    keywords: List[str]
    requires_decomposition: bool
    estimated_chunks_needed: int


@dataclass
class SubQuery:
    """One part of a complex query that was broken down."""
    text: str
    priority: int
    depends_on: List[int]  # Which other sub-queries this needs
    query_type: QueryType
    context_requirements: List[str] = None  # Required context from dependencies
    estimated_cost: float = 0.0  # Estimated processing cost
    confidence_threshold: float = 0.8  # Minimum confidence required
    
    def __post_init__(self):
        if self.context_requirements is None:
            self.context_requirements = []


class QueryClassifier:
    """Intelligent query classification system for routing decisions."""
    
    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("ROUTING_MODEL", "gpt-3.5-turbo")
        self.llm = OpenAI(
            model=self.model,
            temperature=0.1,
            max_tokens=512,
        )
        
        # Pattern-based classification rules for efficiency
        self.patterns = {
            QueryType.FACTUAL: [
                r'\bwhat is\b', r'\bwho is\b', r'\bwhen did\b', r'\bwhere is\b',
                r'\bhow many\b', r'\bdefine\b', r'\blist\b'
            ],
            QueryType.PROCEDURAL: [
                r'\bhow to\b', r'\bsteps to\b', r'\bprocess of\b', r'\bguide\b',
                r'\binstructions\b', r'\btutorial\b'
            ],
            QueryType.COMPARATIVE: [
                r'\bcompare\b', r'\bdifference\b', r'\bversus\b', r'\bvs\b',
                r'\bbetter than\b', r'\bsimilar to\b'
            ],
            QueryType.ANALYTICAL: [
                r'\bwhy\b', r'\banalyze\b', r'\bexplain\b', r'\breason\b',
                r'\bcause\b', r'\bimpact\b', r'\beffect\b'
            ]
        }
    
    async def classify_query(self, query: str) -> QueryClassification:
        """
        Classify query type and characteristics for routing decisions.
        
        Args:
            query: The user query to classify
            
        Returns:
            QueryClassification with routing recommendations
        """
        try:
            # Quick pattern-based classification
            pattern_type, pattern_confidence = self._pattern_classify(query)
            
            # Analyze query complexity
            complexity_score = self._calculate_complexity(query)
            
            # Use LLM for nuanced classification if needed
            if pattern_confidence < 0.8 or complexity_score > 0.7:
                llm_classification = await self._llm_classify(query)
                if llm_classification:
                    query_type, confidence = llm_classification
                else:
                    query_type, confidence = pattern_type, pattern_confidence
            else:
                query_type, confidence = pattern_type, pattern_confidence
            
            # Extract keywords
            keywords = self._extract_keywords(query)
            
            # Determine if decomposition is needed
            requires_decomposition = (
                complexity_score > float(os.getenv("QUERY_COMPLEXITY_THRESHOLD", "0.7")) or
                query_type == QueryType.MULTIFACETED or
                len(self._find_conjunctions(query)) > 1
            )
            
            # Estimate chunks needed
            estimated_chunks = self._estimate_chunks_needed(query, query_type, complexity_score)
            
            classification = QueryClassification(
                query_type=query_type,
                confidence=confidence,
                complexity_score=complexity_score,
                keywords=keywords,
                requires_decomposition=requires_decomposition,
                estimated_chunks_needed=estimated_chunks
            )
            
            logger.info(f"Classified query as {query_type.value} with confidence {confidence:.2f}")
            return classification
            
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            # Fallback classification
            return QueryClassification(
                query_type=QueryType.SEMANTIC,
                confidence=0.5,
                complexity_score=0.5,
                keywords=query.split()[:5],
                requires_decomposition=False,
                estimated_chunks_needed=5
            )
    
    def _pattern_classify(self, query: str) -> Tuple[QueryType, float]:
        """Fast pattern-based classification."""
        query_lower = query.lower()
        scores = {}
        
        for query_type, patterns in self.patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            if score > 0:
                scores[query_type] = score / len(patterns)
        
        if not scores:
            # Check for multi-part questions
            conjunctions = len(self._find_conjunctions(query))
            if conjunctions > 1:
                return QueryType.MULTIFACETED, 0.8
            return QueryType.SEMANTIC, 0.6
        
        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(scores[best_type] + 0.6, 1.0)
        
        return best_type, confidence
    
    async def _llm_classify(self, query: str) -> Optional[Tuple[QueryType, float]]:
        """LLM-based classification for complex queries."""
        try:
            prompt = f"""
Classify this query into one of these categories:
- factual: Direct fact retrieval (what, who, when, where)
- semantic: Complex semantic understanding  
- comparative: Comparison between concepts
- procedural: How-to or step-by-step instructions
- analytical: Analysis or reasoning required
- multifaceted: Complex multi-part questions

Query: "{query}"

Respond with only the category name and a confidence score (0.0-1.0) separated by a comma.
Example: "factual, 0.85"
"""
            
            response = await self.llm.acomplete(prompt)
            result = response.text.strip().lower()
            
            parts = result.split(',')
            if len(parts) == 2:
                category = parts[0].strip()
                confidence = float(parts[1].strip())
                
                try:
                    query_type = QueryType(category)
                    return query_type, confidence
                except ValueError:
                    logger.warning(f"Invalid query type from LLM: {category}")
                    
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            
        return None
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score (0.0-1.0)."""
        factors = []
        
        # Length factor
        word_count = len(query.split())
        length_score = min(word_count / 20.0, 1.0)
        factors.append(length_score * 0.3)
        
        # Conjunction factor (multiple parts)
        conjunctions = len(self._find_conjunctions(query))
        conjunction_score = min(conjunctions / 3.0, 1.0)
        factors.append(conjunction_score * 0.4)
        
        # Question complexity
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        question_count = sum(1 for word in question_words if word in query.lower())
        question_score = min(question_count / 3.0, 1.0)
        factors.append(question_score * 0.3)
        
        return sum(factors)
    
    def _find_conjunctions(self, query: str) -> List[str]:
        """Find conjunctions that indicate multi-part queries."""
        conjunctions = ['and', 'or', 'but', 'also', 'additionally', 'furthermore', 'moreover']
        query_lower = query.lower()
        return [conj for conj in conjunctions if f' {conj} ' in query_lower]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from the query."""
        # Simple keyword extraction - can be enhanced with NLP libraries
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'why', 'when', 'where', 'who', 'which'}
        
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def _estimate_chunks_needed(self, query: str, query_type: QueryType, complexity: float) -> int:
        """Estimate number of chunks needed for this query."""
        base_chunks = {
            QueryType.FACTUAL: 3,
            QueryType.SEMANTIC: 5,
            QueryType.COMPARATIVE: 7,
            QueryType.PROCEDURAL: 6,
            QueryType.ANALYTICAL: 8,
            QueryType.MULTIFACETED: 10
        }
        
        base = base_chunks.get(query_type, 5)
        complexity_multiplier = 1.0 + (complexity * 0.5)
        
        return min(int(base * complexity_multiplier), 15)


class QueryDecomposer:
    """Intelligent query decomposition for complex multi-part questions."""
    
    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("DECOMPOSITION_MODEL", "gpt-3.5-turbo")
        self.llm = OpenAI(
            model=self.model,
            temperature=0.2,
            max_tokens=1024,
        )
        self.max_subqueries = int(os.getenv("MAX_SUBQUERIES", "3"))
        self.max_cost_per_decomposition = float(os.getenv("MAX_DECOMPOSITION_COST", "0.50"))
        self.cost_per_token = 0.0000015  # Approximate cost for gpt-3.5-turbo
        self.decomposition_cache = {}  # Cache for expensive decompositions
        self.success_metrics = {
            'total_decompositions': 0,
            'successful_decompositions': 0,
            'failed_decompositions': 0,
            'cost_saved_via_cache': 0.0,
            'avg_subqueries': 0.0
        }
    
    async def decompose_query(self, query: str, classification: QueryClassification) -> List[SubQuery]:
        """
        Decompose complex query into manageable sub-queries with enhanced dependency management.
        
        Args:
            query: The complex query to decompose
            classification: Query classification information
            
        Returns:
            List of SubQuery objects with dependencies and cost information
        """
        try:
            self.success_metrics['total_decompositions'] += 1
            
            if not classification.requires_decomposition:
                # Return the original query as a single sub-query
                single_query = SubQuery(
                    text=query,
                    priority=1,
                    depends_on=[],
                    query_type=classification.query_type,
                    estimated_cost=self._estimate_query_cost(query)
                )
                return [single_query]
            
            # Try to use IntelligentCacheManager for decomposition caching
            try:
                from src.performance import get_performance_optimizer
                optimizer = get_performance_optimizer()
                cache_manager = optimizer.cache_manager
                
                # Use intelligent cache manager for decomposition
                async def compute_decomposition():
                    return await self._perform_decomposition(query, classification)
                
                subqueries = await cache_manager.get_or_compute_decomposition(
                    query,
                    classification,
                    compute_decomposition
                )
                
                self.success_metrics['successful_decompositions'] += 1
                self.success_metrics['avg_subqueries'] = (
                    (self.success_metrics['avg_subqueries'] * (self.success_metrics['successful_decompositions'] - 1) + len(subqueries)) /
                    self.success_metrics['successful_decompositions']
                )
                
                return subqueries
                
            except ImportError:
                # Fallback to standard caching if performance module not available
                pass
            
            # Check cache first (fallback)
            cache_key = self._generate_cache_key(query, classification)
            if cache_key in self.decomposition_cache:
                logger.info("Using cached decomposition")
                cached_result = self.decomposition_cache[cache_key]
                self.success_metrics['cost_saved_via_cache'] += cached_result['estimated_cost']
                return cached_result['subqueries']
            
            # Perform the actual decomposition
            subqueries = await self._perform_decomposition(query, classification)
            
            # Cache the result (fallback)
            cache_key = self._generate_cache_key(query, classification)
            total_cost = sum(sq.estimated_cost for sq in subqueries)
            self.decomposition_cache[cache_key] = {
                'subqueries': subqueries,
                'estimated_cost': total_cost,
                'timestamp': time.time()
            }
            
            self.success_metrics['successful_decompositions'] += 1
            self.success_metrics['avg_subqueries'] = (
                (self.success_metrics['avg_subqueries'] * (self.success_metrics['successful_decompositions'] - 1) + len(subqueries)) /
                self.success_metrics['successful_decompositions']
            )
            
            logger.info(f"Decomposed query into {len(subqueries)} sub-queries (cost: ${total_cost:.4f})")
            return subqueries
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            self.success_metrics['failed_decompositions'] += 1
            
            # Enhanced fallback: return original query with estimated cost
            fallback_query = SubQuery(
                text=query,
                priority=1,
                depends_on=[],
                query_type=classification.query_type,
                estimated_cost=self._estimate_query_cost(query)
            )
            return [fallback_query]
    
    async def _perform_decomposition(self, query: str, classification: QueryClassification) -> List[SubQuery]:
        """
        Perform the actual query decomposition.
        
        Args:
            query: Query to decompose
            classification: Query classification
            
        Returns:
            List of sub-queries
        """
        # Use LLM to decompose the query with enhanced prompting
        subqueries = await self._llm_decompose_enhanced(query, classification)
        
        # Validate and optimize sub-queries
        subqueries = self._validate_and_optimize_subqueries(subqueries, classification)
        
        # Calculate total cost and apply limits
        total_cost = sum(sq.estimated_cost for sq in subqueries)
        if total_cost > self.max_cost_per_decomposition:
            subqueries = self._apply_cost_optimization(subqueries, self.max_cost_per_decomposition)
            logger.warning(f"Applied cost optimization: reduced cost from ${total_cost:.4f} to ${sum(sq.estimated_cost for sq in subqueries):.4f}")
        
        return subqueries
    
    async def _llm_decompose_enhanced(self, query: str, classification: QueryClassification) -> List[SubQuery]:
        """Use LLM to decompose complex queries with enhanced dependency management."""
        prompt = f"""
Break down this complex query into {min(self.max_subqueries, 3)} simpler, focused sub-queries with intelligent dependency management.

Original Query: "{query}"
Query Type: {classification.query_type.value}
Keywords: {', '.join(classification.keywords[:5])}
Complexity Score: {classification.complexity_score:.2f}
Confidence: {classification.confidence:.2f}

Guidelines:
1. Each sub-query should be self-contained and answerable independently
2. Number sub-queries in order of priority (1 = highest priority)
3. Identify dependencies and required context from other sub-queries
4. Classify each sub-query type: factual, semantic, comparative, procedural, analytical
5. Consider cost optimization - simpler queries are preferred when possible
6. Ensure each sub-query contributes unique value to answering the original question

Format each sub-query as:
Priority: [number]
Text: [sub-query text]
Type: [query type]
Depends on: [comma-separated priority numbers, or 'none']
Context needed: [brief description of what context from dependencies is needed, or 'none']
Confidence threshold: [0.0-1.0, minimum confidence required for this sub-query]

Example:
Priority: 1
Text: What is machine learning?
Type: factual
Depends on: none
Context needed: none
Confidence threshold: 0.8

Priority: 2
Text: How does machine learning differ from traditional programming approaches?
Type: comparative
Depends on: 1
Context needed: definition and key characteristics of machine learning
Confidence threshold: 0.7

Important: Minimize dependencies where possible to enable parallel execution.
"""
        
        try:
            response = await self.llm.acomplete(prompt)
            return self._parse_subqueries_enhanced(response.text)
        except Exception as e:
            logger.error(f"Enhanced LLM decomposition failed: {e}")
            raise
    
    def _parse_subqueries_enhanced(self, response: str) -> List[SubQuery]:
        """Parse LLM response into enhanced SubQuery objects."""
        subqueries = []
        
        # Split response into blocks
        blocks = response.strip().split('\n\n')
        
        for block in blocks:
            if not block.strip():
                continue
                
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            
            # Parse each block with enhanced fields
            priority = 1
            text = ""
            query_type = QueryType.SEMANTIC
            depends_on = []
            context_requirements = []
            confidence_threshold = 0.8
            
            for line in lines:
                if line.lower().startswith('priority:'):
                    try:
                        priority = int(line.split(':', 1)[1].strip())
                    except ValueError:
                        priority = 1
                        
                elif line.lower().startswith('text:'):
                    text = line.split(':', 1)[1].strip()
                    
                elif line.lower().startswith('type:'):
                    type_str = line.split(':', 1)[1].strip().lower()
                    try:
                        query_type = QueryType(type_str)
                    except ValueError:
                        query_type = QueryType.SEMANTIC
                        
                elif line.lower().startswith('depends on:'):
                    deps_str = line.split(':', 1)[1].strip().lower()
                    if deps_str != 'none':
                        try:
                            depends_on = [int(d.strip()) for d in deps_str.split(',') if d.strip().isdigit()]
                        except ValueError:
                            depends_on = []
                            
                elif line.lower().startswith('context needed:'):
                    context_str = line.split(':', 1)[1].strip()
                    if context_str.lower() != 'none':
                        context_requirements = [context_str]
                        
                elif line.lower().startswith('confidence threshold:'):
                    try:
                        confidence_threshold = float(line.split(':', 1)[1].strip())
                        confidence_threshold = max(0.0, min(1.0, confidence_threshold))
                    except ValueError:
                        confidence_threshold = 0.8
            
            if text:
                estimated_cost = self._estimate_query_cost(text)
                subqueries.append(SubQuery(
                    text=text,
                    priority=priority,
                    depends_on=depends_on,
                    query_type=query_type,
                    context_requirements=context_requirements,
                    estimated_cost=estimated_cost,
                    confidence_threshold=confidence_threshold
                ))
        
        # Sort by priority
        subqueries.sort(key=lambda x: x.priority)
        
        # Validate and optimize dependencies
        subqueries = self._validate_dependencies(subqueries)
        
        return subqueries
    
    def _estimate_query_cost(self, query: str) -> float:
        """Estimate the processing cost of a query based on length and complexity."""
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = len(query) / 4
        
        # Add overhead for processing, retrieval, and generation
        total_tokens = estimated_tokens * 3  # Input + retrieval context + output
        
        return total_tokens * self.cost_per_token
    
    def _generate_cache_key(self, query: str, classification: QueryClassification) -> str:
        """Generate cache key for decomposition results."""
        import hashlib
        key_data = f"{query}_{classification.query_type.value}_{classification.complexity_score:.2f}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _validate_and_optimize_subqueries(self, subqueries: List[SubQuery], classification: QueryClassification) -> List[SubQuery]:
        """Validate and optimize the list of sub-queries."""
        if not subqueries:
            return subqueries
            
        # Remove duplicates based on text similarity
        unique_subqueries = self._remove_duplicate_subqueries(subqueries)
        
        # Limit number of sub-queries
        if len(unique_subqueries) > self.max_subqueries:
            # Keep the highest priority queries
            unique_subqueries = sorted(unique_subqueries, key=lambda x: x.priority)[:self.max_subqueries]
            logger.warning(f"Limited sub-queries to {self.max_subqueries}")
        
        # Validate dependency chains
        unique_subqueries = self._validate_dependencies(unique_subqueries)
        
        return unique_subqueries
    
    def _remove_duplicate_subqueries(self, subqueries: List[SubQuery]) -> List[SubQuery]:
        """Remove duplicate or highly similar sub-queries."""
        unique_queries = []
        
        for query in subqueries:
            is_duplicate = False
            for existing in unique_queries:
                # Simple similarity check based on common words
                query_words = set(query.text.lower().split())
                existing_words = set(existing.text.lower().split())
                
                if query_words and existing_words:
                    intersection = query_words.intersection(existing_words)
                    union = query_words.union(existing_words)
                    similarity = len(intersection) / len(union)
                    
                    if similarity > 0.8:  # High similarity threshold
                        is_duplicate = True
                        logger.info(f"Removed duplicate sub-query: {query.text}")
                        break
            
            if not is_duplicate:
                unique_queries.append(query)
        
        return unique_queries
    
    def _validate_dependencies(self, subqueries: List[SubQuery]) -> List[SubQuery]:
        """Validate and optimize dependency relationships."""
        valid_priorities = {sq.priority for sq in subqueries}
        
        for sq in subqueries:
            # Remove invalid dependencies
            sq.depends_on = [dep for dep in sq.depends_on if dep in valid_priorities and dep != sq.priority]
            
            # Detect and resolve circular dependencies
            if self._has_circular_dependency(sq, subqueries):
                logger.warning(f"Circular dependency detected for sub-query {sq.priority}, removing dependencies")
                sq.depends_on = []
        
        # Optimize for parallel execution by reducing unnecessary dependencies
        subqueries = self._optimize_dependencies_for_parallelism(subqueries)
        
        return subqueries
    
    def _has_circular_dependency(self, subquery: SubQuery, all_subqueries: List[SubQuery]) -> bool:
        """Check if a sub-query has circular dependencies."""
        visited = set()
        
        def check_circular(current_priority: int, path: List[int]) -> bool:
            if current_priority in path:
                return True
            
            path.append(current_priority)
            
            # Find sub-query with this priority
            current_sq = next((sq for sq in all_subqueries if sq.priority == current_priority), None)
            if not current_sq:
                return False
            
            for dep in current_sq.depends_on:
                if check_circular(dep, path.copy()):
                    return True
            
            return False
        
        return check_circular(subquery.priority, [])
    
    def _optimize_dependencies_for_parallelism(self, subqueries: List[SubQuery]) -> List[SubQuery]:
        """Optimize dependencies to maximize parallel execution opportunities."""
        # For queries that don't actually need context from dependencies, remove the dependency
        for sq in subqueries:
            if sq.depends_on and not sq.context_requirements:
                # If no specific context is needed, remove dependencies for parallel execution
                logger.info(f"Removing unnecessary dependencies for sub-query {sq.priority} to enable parallel execution")
                sq.depends_on = []
        
        return subqueries
    
    def _apply_cost_optimization(self, subqueries: List[SubQuery], max_cost: float) -> List[SubQuery]:
        """Apply cost optimization by reducing or simplifying sub-queries."""
        # Sort by cost descending
        sorted_queries = sorted(subqueries, key=lambda x: x.estimated_cost, reverse=True)
        optimized_queries = []
        total_cost = 0.0
        
        for sq in sorted_queries:
            if total_cost + sq.estimated_cost <= max_cost:
                optimized_queries.append(sq)
                total_cost += sq.estimated_cost
            else:
                # Try to simplify the query to fit within budget
                simplified_query = self._simplify_subquery(sq, max_cost - total_cost)
                if simplified_query and simplified_query.estimated_cost > 0:
                    optimized_queries.append(simplified_query)
                    total_cost += simplified_query.estimated_cost
                else:
                    logger.warning(f"Dropped sub-query due to cost constraints: {sq.text}")
        
        # Sort back by priority
        optimized_queries.sort(key=lambda x: x.priority)
        return optimized_queries
    
    def _simplify_subquery(self, subquery: SubQuery, available_budget: float) -> Optional[SubQuery]:
        """Simplify a sub-query to fit within the available budget."""
        if available_budget <= 0:
            return None
        
        # Simple approach: truncate the query text if it's too long
        if len(subquery.text) > 100 and available_budget < subquery.estimated_cost:
            # Truncate to fit budget (rough approximation)
            max_length = int(100 * (available_budget / subquery.estimated_cost))
            simplified_text = subquery.text[:max_length] + "..."
            
            simplified_query = SubQuery(
                text=simplified_text,
                priority=subquery.priority,
                depends_on=subquery.depends_on,
                query_type=subquery.query_type,
                context_requirements=subquery.context_requirements,
                estimated_cost=self._estimate_query_cost(simplified_text),
                confidence_threshold=max(0.6, subquery.confidence_threshold - 0.1)  # Lower threshold for simplified query
            )
            
            logger.info(f"Simplified sub-query {subquery.priority}: reduced cost from ${subquery.estimated_cost:.4f} to ${simplified_query.estimated_cost:.4f}")
            return simplified_query
        
        return subquery
    
    def get_decomposition_metrics(self) -> Dict[str, Any]:
        """Get metrics about decomposition performance."""
        total = self.success_metrics['total_decompositions']
        if total == 0:
            return {"total_decompositions": 0, "success_rate": 0.0}
        
        return {
            "total_decompositions": total,
            "successful_decompositions": self.success_metrics['successful_decompositions'],
            "failed_decompositions": self.success_metrics['failed_decompositions'],
            "success_rate": self.success_metrics['successful_decompositions'] / total,
            "average_subqueries_per_decomposition": self.success_metrics['avg_subqueries'],
            "cost_saved_via_cache": self.success_metrics['cost_saved_via_cache'],
            "cache_size": len(self.decomposition_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear the decomposition cache."""
        self.decomposition_cache.clear()
        logger.info("Decomposition cache cleared")