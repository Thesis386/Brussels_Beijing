"""Reviewed RQ5 framing constructs built on top of the raw EN/ZH lexicons.

The raw dictionaries in ``nb5d_keyword_framing.py`` and ``zh_framing_terms.py`` are
deliberately expansive discovery inventories.  This module defines the smaller set of
response-level constructs used for analysis.  It keeps four conceptually distinct
dimensions separate:

* frames: values or interpretive lenses used to characterize an issue;
* actors (SALIENCE): which actors the response makes prominent by mentioning them.  This
  is deliberately a salience measure, not an agency measure -- a match means the actor is
  named, not that the actor is the causal agent of the action.  Who *acts* (agent vs.
  affected vs. opponent) is coded only in the manual close reading, where syntax and
  context are read directly;
* mechanisms: the governance intervention proposed or discussed;
* risk types: the concrete harm used to justify a position.

The frame constructs aggregate existing raw-term labels.  Actor, mechanism, and risk
patterns are matched directly because the discovery lexicons were not designed to
represent those dimensions comprehensively.  All constructs are multi-label: a response
can invoke several at once.  This is a draft measurement instrument pending human review
of the validation workbook produced by ``nb5i_stance_conditioned_framing.py``.
"""

from __future__ import annotations


# Only constructs with a defensible counterpart in both languages belong here.  Values
# are raw-term labels as they appear in the corresponding language dictionary.
FRAME_COMPONENTS = {
    # Split from a former "safety/catastrophic risk" frame: the frontier loss-of-control
    # lens is a distinct moral evaluation from ordinary product safety, and merging them
    # was noisy given the polysemy of Chinese 安全 (safety/security).
    "catastrophic/existential risk": {
        "en": ["catastrophic", "existential", "high/systemic risk"],
        "zh": ["catastrophic", "existential", "high/systemic risk"],
    },
    "everyday safety/reliability": {
        "en": ["safety"],
        "zh": ["safety"],
    },
    # Split from a former "rights/dignity/privacy" frame: the constitutional human-rights
    # register vs. the concrete data-subject / consumer-protection register (GDPR lineage).
    "fundamental rights/dignity": {
        "en": ["human rights", "fundamental rights/dignity", "civil rights/liberties"],
        "zh": ["human rights", "fundamental rights/dignity", "civil rights/liberties"],
    },
    "privacy/consumer protection": {
        "en": ["privacy", "consumer protection"],
        "zh": ["privacy", "consumer protection"],
    },
    # Split from a former "innovation/competitiveness" frame: a market-liberal "do not
    # stifle growth" lens vs. a nationalist "do not cede the race" leadership lens. These
    # are the origin-linked framings RQ5 is about, so they must not collapse together.
    "innovation/economic dynamism": {
        "en": ["innovation", "responsible innovation"],
        "zh": ["innovation", "innovation vitality (创新活力)"],
    },
    "competitiveness/tech leadership": {
        "en": ["competitiveness", "AI dominance/leadership"],
        "zh": ["competitiveness", "AI dominance/leadership"],
    },
    "national security": {
        "en": ["national security", "foreign adversaries"],
        "zh": ["national security", "foreign adversaries"],
    },
    "geopolitical competition/restriction": {
        "en": [
            "arms race", "competition/race", "strategic competition", "containment",
            "decoupling", "chokepoint", "small yard high fence", "technology blockade",
        ],
        "zh": [
            "arms race", "competition/race", "strategic competition", "containment",
            "decoupling", "chokepoint", "small yard high fence",
            "tech blockade (技术封锁)",
        ],
    },
    "sovereignty/state capacity": {
        "en": ["sovereignty", "data sovereignty/localization", "state-led development"],
        "zh": [
            "sovereignty", "data sovereignty/localization", "state-led development",
            "safe and controllable (安全可控)", "tech self-reliance (自立自强)",
        ],
    },
    "social stability/collective good": {
        "en": ["social stability/collective benefit", "core socialist values"],
        "zh": ["social stability/collective benefit", "core socialist values"],
    },
    "transparency/accountability": {
        "en": ["transparency", "accountability", "developer responsibility"],
        "zh": ["transparency", "accountability", "developer responsibility"],
    },
    "openness/access/inclusion": {
        "en": [
            "open source", "open weights", "developing countries/Global South",
            "digital divide", "inclusive and mutually beneficial",
        ],
        "zh": [
            "open source", "open weights", "developing countries advocacy (发展中国家)",
            "digital divide (数字鸿沟)", "inclusive / mutual benefit (普惠/互利共赢)",
        ],
    },
    "multilateral cooperation": {
        "en": [
            "United Nations/UN", "global governance", "international standards",
            "true multilateralism",
        ],
        "zh": [
            "United Nations/UN", "global governance (全球治理)", "international standards",
            "international cooperation (国际合作)", "true multilateralism (真正的多边主义)",
            "multistakeholder (多利益攸关方)",
        ],
    },
    "regulatory flexibility/voluntarism": {
        "en": ["deregulation", "voluntary/self-regulation", "regulatory sandbox"],
        "zh": [
            "deregulation", "voluntary/self-regulation", "regulatory sandbox (监管沙盒)",
            "agile governance (敏捷治理)",
        ],
    },
    "precautionary/binding oversight": {
        "en": [
            "ex-ante / pre-deployment", "risk-based regulation", "conformity assessment",
            "market surveillance",
        ],
        "zh": [
            "ex-ante / pre-deployment", "risk-based regulation", "conformity assessment",
            "market surveillance",
        ],
    },
}


# Direct bilingual patterns.  English is compiled case-insensitively; Chinese is matched
# without word boundaries.  Patterns favor precision over recall for the first review.
ACTOR_PATTERNS = {
    "government/regulator": {
        "en": r"\b(?:government|regulator|regulatory authorit(?:y|ies)|public authorit(?:y|ies)|national authorit(?:y|ies)|local authorit(?:y|ies)|Chinese authorities|European authorities|US authorities|U\.S\. authorities|legislator|lawmakers?|state agenc(?:y|ies))\b",
        "zh": r"政府|监管机构|监管部门|主管部门|国家机关|国家有关部门|地方当局|国家当局|中国当局|欧盟当局|美国当局|立法者|立法机构",
    },
    "law enforcement/security agencies": {
        "en": r"\b(?:law enforcement(?: agenc(?:y|ies))?|police|police forces?|security services?|security agenc(?:y|ies)|intelligence services?|intelligence agenc(?:y|ies))\b",
        "zh": r"执法机构|执法部门|公安机关|公安部门|警察|警方|安全机构|安全部门|情报机构|情报部门",
    },
    "courts/judiciary": {
        "en": r"\b(?:courts?|the judiciary|judicial (?:system|review|oversight|authorit(?:y|ies))|judges?|tribunals?|supreme court|constitutional court)\b",
        "zh": r"法院|司法机构|司法机关|司法系统|司法审查|法官|法庭|最高法院|宪法法院|审判机关",
    },
    "foreign governments/states": {
        "en": r"\b(?:foreign governments?|foreign states?|foreign countries|other governments?|third-country governments?|rival governments?)\b",
        "zh": r"外国政府|外国当局|外国国家|其他国家政府|他国政府|第三国政府|竞争对手政府",
    },
    "AI developer/provider": {
        "en": r"\b(?:AI|artificial intelligence|model|technology|tech) (?:developer|provider|compan(?:y|ies)|firm|lab)s?\b|\bmodel providers?\b",
        "zh": r"人工智能公司|AI公司|科技公司|科技企业|模型开发者|模型提供商|人工智能开发者|AI开发者",
    },
    "public/citizens/users": {
        "en": r"\b(?:the public|citizens?|ordinary people|members of the public|consumers?|end[- ]users?|users?)\b",
        "zh": r"公众|公民|普通民众|社会大众|消费者|最终用户|终端用户|用户",
    },
    "researchers/academia": {
        "en": r"\b(?:researchers?|scientists?|academics?|academia|universit(?:y|ies)|research institutions?)\b",
        "zh": r"研究人员|科研人员|科学家|学者|学术界|高校|大学|研究机构|科研机构",
    },
    "civil society/rights advocates": {
        "en": r"\b(?:civil society|non[- ]governmental organi[sz]ations?|NGOs?|public[- ]interest groups?|human rights organi[sz]ations?|rights groups?|privacy advocates?|civil liberties groups?|watchdog groups?)\b",
        "zh": r"民间社会|社会组织|非政府组织|公益组织|公共利益组织|人权组织|权利组织|隐私倡导者|公民自由组织|监督组织",
    },
    "news media/press": {
        "en": r"\b(?:the (?:news )?media|the press|press freedom|journalists?|news outlets?|news organi[sz]ations?|newspapers?|reporters?)\b",
        "zh": r"新闻媒体|大众媒体|新闻界|新闻机构|新闻报道|新闻自由|记者|报社|媒体监督",
    },
    "workers/creators": {
        "en": r"\b(?:workers?|employees?|labor|labour|artists?|authors?|writers?|creators?|content creators?)\b",
        "zh": r"劳动者|工人|员工|从业者|艺术家|作者|作家|创作者|内容创作者",
    },
    "international organizations": {
        "en": r"\b(?:international organi[sz]ations?|multilateral institutions?|United Nations|OECD|G7|UNESCO)\b",
        "zh": r"国际组织|多边机构|联合国|经合组织|经济合作与发展组织|七国集团|教科文组织",
    },
    "developing countries/Global South": {
        "en": r"\b(?:developing countries|developing economies|developing nations|Global South)\b",
        "zh": r"发展中国家|发展中经济体|全球南方",
    },
}


MECHANISM_PATTERNS = {
    "licensing/authorization": {
        "en": r"\b(?:licen[cs](?:e|ing|ure)|government authorization|prior authorization|preclearance|permitting requirement|approval requirement)\b",
        "zh": r"许可(?:证|制度)?|牌照|政府授权|预先授权|事先授权|审批制度|审批要求|准入许可",
    },
    "registration/filing": {
        "en": r"\b(?:registration|registering|filing requirement|mandatory filing)\b",
        "zh": r"登记制度|注册制度|备案制度|登记要求|注册要求|备案要求|强制备案",
    },
    "audit/evaluation/testing": {
        "en": r"\b(?:independent audit|external audit|algorithmic audit|safety evaluation|risk assessment|red[- ]team(?:ing)?|safety testing|capability evaluation)s?\b",
        "zh": r"独立审计|外部审计|算法审计|安全评估|风险评估|红队测试|安全测试|能力评估|第三方评估",
    },
    "disclosure/reporting": {
        "en": r"\b(?:disclosure requirements?|reporting requirements?|mandatory disclosure|transparency reports?|public reporting)\b",
        "zh": r"披露要求|报告义务|报告要求|强制披露|透明度报告|公开报告|信息披露",
    },
    "liability/compensation": {
        "en": r"\b(?:legal liability|civil liability|strict liability|product liability|compensation|damages|right to redress)\b",
        "zh": r"法律责任|民事责任|严格责任|产品责任|赔偿责任|损害赔偿|救济权|追究责任",
    },
    "prohibition/moratorium": {
        "en": r"\b(?:prohibition|ban|moratorium|temporary pause|development pause|deployment pause)s?\b",
        "zh": r"禁止|禁令|暂停令|暂缓|暂停开发|暂停部署|开发禁令|部署禁令",
    },
    "enforcement/penalties": {
        "en": r"\b(?:enforcement|administrative penalt(?:y|ies)|criminal penalt(?:y|ies)|fines?|sanctions?)\b",
        "zh": r"执法|行政处罚|刑事处罚|罚款|制裁|处罚措施|惩罚措施",
    },
    "technical safeguards/access controls": {
        "en": r"\b(?:technical safeguards?|safety measures?|protective measures?|access controls?|usage restrictions?|structured access)\b",
        "zh": r"技术保障|技术防护|安全措施|防护措施|访问控制|使用限制|结构化访问|分级访问",
    },
    "voluntary codes/commitments": {
        "en": r"\b(?:voluntary commitments?|voluntary codes?|codes? of conduct|industry standards?|industry self[- ]regulation)\b",
        "zh": r"自愿承诺|自愿准则|行为准则|行业规范|行业标准|行业自律|自我监管",
    },
    "international standards/coordination": {
        "en": r"\b(?:international standards?|global standards?|international coordination|multilateral coordination|regulatory cooperation)\b",
        "zh": r"国际标准|全球标准|国际协调|多边协调|监管合作|国际监管合作",
    },
}


RISK_PATTERNS = {
    "cybersecurity/malware": {
        "en": r"\b(?:cybersecurity|cyberattacks?|cyber attacks?|malware|ransomware|hacking|computer intrusion)s?\b",
        "zh": r"网络安全|网络攻击|恶意软件|勒索软件|黑客攻击|计算机入侵",
    },
    "misinformation/manipulation": {
        "en": r"\b(?:misinformation|disinformation|fake news|information manipulation|influence operations?|propaganda)\b",
        "zh": r"虚假信息|错误信息|假新闻|信息操纵|舆论操纵|影响力行动|宣传操纵",
    },
    "CBRN/biological misuse": {
        "en": r"\b(?:CBRN|biological weapons?|chemical weapons?|nuclear weapons?|weapons? of mass destruction|pathogens?|biological misuse|biosecurity)\b",
        "zh": r"生化核放|生物武器|化学武器|核武器|大规模杀伤性武器|病原体|生物滥用|生物安全",
    },
    "bias/discrimination": {
        "en": r"\b(?:algorithmic bias|discrimination|discriminatory|unfair bias|racial bias|gender bias)\b",
        "zh": r"算法偏见|算法偏差|歧视|歧视性|不公平偏见|种族偏见|性别偏见",
    },
    "labor displacement": {
        "en": r"\b(?:job displacement|job losses|labor displacement|labour displacement|technological unemployment|workers? replaced|replace jobs)\b",
        "zh": r"岗位流失|工作岗位流失|就业替代|劳动力替代|技术性失业|取代工人|取代工作",
    },
    "concentration/monopoly": {
        "en": r"\b(?:market concentration|concentration of power|monopoly|monopolies|oligopoly|corporate dominance)\b",
        "zh": r"市场集中|权力集中|垄断|寡头|企业主导|公司支配",
    },
    "malicious use/crime": {
        "en": r"\b(?:malicious use|misuse|criminal use|illegal use|fraud|scams?|abuse by bad actors?)\b",
        "zh": r"恶意使用|恶意用途|滥用|犯罪用途|非法用途|欺诈|诈骗|不法分子滥用",
    },
    "irreversibility/proliferation": {
        "en": r"\b(?:irreversible|cannot be recalled|cannot be taken back|uncontrolled proliferation|capability proliferation|model proliferation|irreversible diffusion)\b",
        "zh": r"不可逆|无法撤回|无法收回|能力扩散|模型扩散|不可控扩散|一旦.{0,8}(?:发布|公开).{0,8}(?:无法|难以)(?:收回|撤回)",
    },
}


FRAME_DEFINITIONS = {
    "catastrophic/existential risk": "The issue is evaluated through systemic, catastrophic, existential, or loss-of-control risk from advanced AI.",
    "everyday safety/reliability": "The issue is evaluated through ordinary product safety, reliability, or harm prevention, without a catastrophic register.",
    "fundamental rights/dignity": "The issue is evaluated through human rights, civil liberties, or human dignity.",
    "privacy/consumer protection": "The issue is evaluated through privacy, data protection, or protection of the individual as a consumer.",
    "innovation/economic dynamism": "The issue is evaluated through innovation and economic dynamism, typically warning against stifling growth.",
    "competitiveness/tech leadership": "The issue is evaluated through national competitiveness or the imperative not to cede technological leadership.",
    "national security": "The issue is evaluated through national security or threats from foreign adversaries.",
    "geopolitical competition/restriction": "The issue is characterized as interstate technological competition, containment, decoupling, or restriction.",
    "sovereignty/state capacity": "The issue is evaluated through sovereignty, localization, state capacity, or technological self-reliance.",
    "social stability/collective good": "The issue is evaluated through social stability, collective welfare, public interest, or socialist values.",
    "transparency/accountability": "The issue is evaluated through transparency, answerability, or developer responsibility.",
    "openness/access/inclusion": "The issue is evaluated through open access, distributed participation, or inclusion of less-resourced actors.",
    "multilateral cooperation": "The issue is evaluated through international institutions, common standards, or multilateral coordination.",
    "regulatory flexibility/voluntarism": "The issue is evaluated through regulatory restraint, flexible governance, sandboxes, or voluntary commitments.",
    "precautionary/binding oversight": "The issue is evaluated through binding, risk-based, or pre-deployment oversight.",
}


def all_construct_names() -> dict[str, list[str]]:
    """Return the ordered construct names by analytical dimension."""
    return {
        "frame": list(FRAME_COMPONENTS),
        "actor": list(ACTOR_PATTERNS),
        "mechanism": list(MECHANISM_PATTERNS),
        "risk_type": list(RISK_PATTERNS),
    }
