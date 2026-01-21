import re
import logging
from typing import Tuple, List

from src.evaluation.data_classes import MetricResult
from src.evaluation.structure import BaseMetric

logger = logging.getLogger(__name__)

class FaithfulnessMetric(BaseMetric):
    """
    Faithfulness: the extent to which the answer is based ONLY on the provided context.

    Important: Doesn't depend on the question or the expected answer.
    Checks that each statement in the answer is supported by the context.
    """

    def __init__(self):
        super().__init__("faithfulness")

    async def calculate(
        self,
        actual: str,
        expected: str,
        context: List[str],
        question: str
    ) -> MetricResult:
        if not context:
            return MetricResult(
                score=0.0,
                reason="Нет контекста для проверки faithfulness",
                details={"error": "empty_context"}
            )


        clean_answer = self._remove_think_tags(actual)
        if not clean_answer.strip():
            return MetricResult(
                score=0.0,
                reason="Ответ пуст после очистки",
                details={"error": "empty_cleaned_answer"}
            )
            
        claims = self._extract_claims(clean_answer)
        if not claims:
            return MetricResult(
                score=1.0,
                reason="В ответе нет проверяемых утверждений",
                details={"claims_count": 0}
            )

        verified_count = 0
        verification_details = []

        for claim in claims:
            is_supported, evidence = await self._is_claim_supported_by_context(claim, context)
            if is_supported:
                verified_count += 1

            verification_details.append({
                "claim": claim[:150] + ("..." if len(claim) > 150 else ""),
                "supported": is_supported,
                "evidence": evidence
            })

        score = verified_count / len(claims)
        details = {
            "total_claims": len(claims),
            "supported_claims": verified_count,
            "verification_details": verification_details
        }

        reason = self._generate_reason(score, len(context), verified_count, len(claims))
        return MetricResult(
            score=round(score, 3),
            reason=reason,
            details=details
        )


    def _remove_think_tags(self, text: str) -> str:
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()


    def _extract_claims(self, text: str) -> List[str]:
        """Smart assertion extraction - ignore logical inferences"""
        clean_text = self._remove_think_tags(text)
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)


        claims = []
        for sent in sentences:
            # Skip
            if any(phrase in sent.lower() for phrase in [
                'эти действия обеспечивают', 'это позволит создать',
                'убедившись что', 'соответствует требованиям'
            ]):
                continue

            if len(sent) > 20 and any(keyword in sent.lower() for keyword in [
                'откройте', 'выберите', 'установите', 'нажмите', 'сохраните'
            ]):

                claims.append(sent)
        return claims


    async def _is_claim_supported_by_context(self, claim: str, context: List[str]) -> Tuple[bool, str]:
        """More flexible assertion support checking"""
        claim_lower = claim.lower().strip()
        if not claim_lower:
            return False, "Пустое утверждение"

        # 1. Checking keywords in context
        claim_keywords = set(re.findall(r'\b[а-яёa-z]{4,}\b', claim_lower))
        if claim_keywords:
            for i, chunk in enumerate(context):
                chunk_lower = chunk.lower()
                # If there is significant overlap of keywords
                chunk_keywords = set(re.findall(r'\b[а-яёa-z]{4,}\b', chunk_lower))
                overlap = claim_keywords & chunk_keywords
                if len(overlap) >= max(2, len(claim_keywords) * 0.3):
                    return True, f"Пересечение ключевых слов в фрагменте {i+1}"


        # 2. LLM test with a more flexible prompt
        try:
            full_context = "\n\n".join(context)
            if len(full_context) > 3000:
                full_context = full_context[:3000] + "..."

            prompt = f"""Утверждение: "{claim}"
            Контекст: {full_context}
            Вопрос: Подтверждается ли утверждение приведенным контекстом? Утверждение может быть парафразом или следствием информации из контекста.
            Ответь "ДА" или "НЕТ".""".strip()

            response = await self._llm_generate(prompt, max_tokens=10)
            clean_resp = response.strip().upper()
            

            if "ДА" in clean_resp:
                return True, "LLM: подтверждено"

            else:
                return False, "LLM: не подтверждено"

        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return False, "Ошибка проверки"


    def _generate_reason(self, score: float, context_len: int, supported: int, total: int) -> str:
        if total == 0:
            return "Нет утверждений для проверки"
        base = (
            "Все утверждения подтверждены контекстом" if score >= 0.95 else
            "Большинство утверждений подтверждено" if score >= 0.7 else
            "Часть утверждений не подтверждена" if score >= 0.3 else
            "Ответ содержит недостоверную информацию"
        )

        return f"{base} ({supported}/{total} утверждений; {context_len} фрагментов контекста)"