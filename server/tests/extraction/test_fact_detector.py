"""Tests for FactDetector — Layer 1 regex-based fact detection."""

import pytest

from app.extraction.fact_detector import FactDetector


@pytest.fixture
def detector():
    return FactDetector()


# ── Chinese patterns ────────────────────────────────────────────

class TestChinesePatterns:
    """Chinese regex patterns for fact detection."""

    def test_chinese_identity_is(self, detector):
        """'我是' pattern — identity statement."""
        assert detector.should_extract("我是张军") is True

    def test_chinese_identity_at(self, detector):
        """'我在' pattern — location/status."""
        assert detector.should_extract("我在北京工作") is True

    def test_chinese_have(self, detector):
        """'我有' pattern — possession."""
        assert detector.should_extract("我有一只猫") is True

    def test_chinese_use(self, detector):
        """'我用' pattern — tool/preference."""
        assert detector.should_extract("我用Python写代码") is True

    def test_chinese_like(self, detector):
        """'我喜欢' pattern — preference."""
        assert detector.should_extract("我喜欢TypeScript") is True

    def test_chinese_prefer(self, detector):
        """'我偏好' pattern — preference."""
        assert detector.should_extract("我偏好简洁的设计") is True

    def test_chinese_moved(self, detector):
        """'我搬到' pattern — location change."""
        assert detector.should_extract("我搬到北京了") is True

    def test_chinese_switched(self, detector):
        """'我换' pattern — change."""
        assert detector.should_extract("我换了一个新手机") is True

    def test_chinese_changed(self, detector):
        """'我改' pattern — change."""
        assert detector.should_extract("我改用VS Code了") is True

    def test_chinese_we_use(self, detector):
        """'我们用' pattern — group decision."""
        assert detector.should_extract("我们用FastAPI做后端") is True

    def test_chinese_we_chose(self, detector):
        """'我们选' pattern — group decision."""
        assert detector.should_extract("我们选了React作为前端框架") is True

    def test_chinese_we_decided(self, detector):
        """'我们决定' pattern — group decision."""
        assert detector.should_extract("我们决定迁移到Kubernetes") is True

    def test_chinese_we_planned(self, detector):
        """'我们计划' pattern — plan."""
        assert detector.should_extract("我们计划下周发布新版本") is True

    def test_chinese_recommend_keyword(self, detector):
        """'推荐' keyword."""
        assert detector.should_extract("有没有推荐的IDE？") is True

    def test_chinese_suggest_keyword(self, detector):
        """'建议' keyword."""
        assert detector.should_extract("我建议用Docker部署") is True

    def test_chinese_preference_keyword(self, detector):
        """'偏好' keyword."""
        assert detector.should_extract("用户的偏好是暗色主题") is True

    def test_chinese_habit_keyword(self, detector):
        """'习惯' keyword."""
        assert detector.should_extract("我的习惯是早上写代码") is True


# ── English patterns ────────────────────────────────────────────

class TestEnglishPatterns:
    """English regex patterns for fact detection."""

    def test_english_i_am(self, detector):
        assert detector.should_extract("I am a software engineer") is True

    def test_english_i_work(self, detector):
        assert detector.should_extract("I work at Google") is True

    def test_english_i_live(self, detector):
        assert detector.should_extract("I live in Tokyo") is True

    def test_english_i_use(self, detector):
        assert detector.should_extract("I use VS Code for development") is True

    def test_english_i_like(self, detector):
        assert detector.should_extract("I like Python") is True

    def test_english_i_prefer(self, detector):
        assert detector.should_extract("I prefer dark mode") is True

    def test_english_i_moved(self, detector):
        assert detector.should_extract("I moved to Berlin") is True

    def test_english_i_switched(self, detector):
        assert detector.should_extract("I switched to Vim") is True

    def test_english_i_changed(self, detector):
        assert detector.should_extract("I changed my editor") is True

    def test_english_we_use(self, detector):
        assert detector.should_extract("We use Kubernetes") is True

    def test_english_we_chose(self, detector):
        assert detector.should_extract("We chose PostgreSQL") is True

    def test_english_we_decided(self, detector):
        assert detector.should_extract("We decided to migrate") is True

    def test_english_we_plan(self, detector):
        assert detector.should_extract("We plan to launch next week") is True

    def test_english_recommend_keyword(self, detector):
        assert detector.should_extract("Any recommendations for a good IDE?") is True

    def test_english_suggest_keyword(self, detector):
        assert detector.should_extract("I suggest using Docker") is True

    def test_english_preference_keyword(self, detector):
        assert detector.should_extract("My preference is light theme") is True

    def test_english_habit_keyword(self, detector):
        assert detector.should_extract("My habit is to code at night") is True

    def test_english_i_have(self, detector):
        assert detector.should_extract("I have a cat named Whiskers") is True

    def test_english_my_name_is(self, detector):
        assert detector.should_extract("My name is Alice") is True

    def test_english_my_job_is(self, detector):
        assert detector.should_extract("My job is frontend development") is True


# ── Negative cases — casual chat should be filtered ─────────────

class TestNegativeCases:
    """Messages that should NOT trigger extraction."""

    def test_simple_greeting(self, detector):
        assert detector.should_extract("你好") is False

    def test_english_greeting(self, detector):
        assert detector.should_extract("Hello") is False

    def test_casual_thanks(self, detector):
        assert detector.should_extract("谢谢") is False

    def test_english_thanks(self, detector):
        assert detector.should_extract("Thanks!") is False

    def test_acknowledgment(self, detector):
        assert detector.should_extract("好的，知道了") is False

    def test_english_ok(self, detector):
        assert detector.should_extract("OK, got it") is False

    def test_question_without_fact(self, detector):
        assert detector.should_extract("今天天气怎么样？") is False

    def test_english_weather_question(self, detector):
        assert detector.should_extract("How's the weather?") is False

    def test_empty_string(self, detector):
        assert detector.should_extract("") is False

    def test_whitespace_only(self, detector):
        assert detector.should_extract("   ") is False

    def test_emoji_only(self, detector):
        assert detector.should_extract("👍") is False


# ── Context-aware boosting ───────────────────────────────────────

class TestContextBoost:
    """Context list can boost detection when message is ambiguous."""

    def test_no_context_pure_question(self, detector):
        assert detector.should_extract("这个怎么用？") is False

    def test_context_boosts_ambiguous(self, detector):
        """When recent context contains fact-like statements,
        even ambiguous follow-ups should be extracted."""
        ctx = ["我最近在学Rust", "Rust的借用检查器挺难的"]
        assert detector.should_extract("这个怎么用？", context=ctx) is True

    def test_context_empty_list_no_boost(self, detector):
        assert detector.should_extract("这个怎么用？", context=[]) is False

    def test_context_non_fact_no_boost(self, detector):
        ctx = ["你好", "谢谢"]
        assert detector.should_extract("这个怎么用？", context=[]) is False


# ── Edge cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_mixed_chinese_english(self, detector):
        """Mixed language message should still match."""
        assert detector.should_extract("我用Next.js做前端") is True

    def test_long_message_with_fact(self, detector):
        """Long message containing a fact pattern."""
        msg = "今天开了三个会，不过我还是决定用Go重写那个服务"
        assert detector.should_extract(msg) is True

    def test_fact_at_end_of_message(self, detector):
        assert detector.should_extract("天气不错，我搬到了深圳") is True