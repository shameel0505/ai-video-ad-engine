from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class BrandKit(BaseModel):
    brand_name: str
    primary_color: str = "#000000"
    secondary_color: str = "#FFFFFF"
    accent_color: str = "#38BDF8"
    font_family: str = "Montserrat-Bold"
    logo_path: Optional[str] = None
    tone_of_voice: str = "modern, punchy, premium"
    forbidden_claims: List[str] = Field(default_factory=list)

class AssetQualityAudit(BaseModel):
    media_type: Literal["photo", "video"]
    subject_detected: str
    lighting_condition: str
    quality_score: float = Field(ge=0.0, le=10.0)
    flaws: List[str] = Field(default_factory=list)
    recommended_fixes: List[str] = Field(default_factory=list)
    raw_file_path: str

class TransitionDirective(BaseModel):
    transition_type: Literal[
        "match_cut", 
        "whip_pan_right", 
        "whip_pan_left", 
        "push_through", 
        "object_wipe", 
        "cross_dissolve", 
        "hard_cut"
    ] = "match_cut"
    overlap_duration_ms: int = 300
    sfx_bridge: Optional[str] = "whoosh_subtle.wav"
    lead_in_ms: int = 250
    camera_exit_motion: str
    camera_entry_motion: str

class SceneManifest(BaseModel):
    scene_id: int
    duration_sec: float = 4.0
    scene_type: Literal["hook", "problem_agitation", "solution_reveal", "feature_deep_dive", "call_to_action"]
    
    # Visual Directives (For Kling / Luma / Runway / FLUX)
    visual_prompt: str
    negative_prompt: str = "blurry, morphing, warped logo, low-res, cartoon, distorted text"
    camera_motion: str
    seed_world_anchor: str
    
    # Continuity & Transitions
    start_frame_source: Literal["raw_asset", "previous_tail_frame", "text_to_video"] = "previous_tail_frame"
    transition_to_next: Optional[TransitionDirective] = None
    
    # Audio & Typography
    on_screen_text: Optional[str] = None
    audio_mode: Literal["voiceover", "music_text_only", "asmr_foley", "original_audio"] = "music_text_only"
    voiceover_script: Optional[str] = None
    audio_sfx: Optional[str] = None

class MasterProductionManifest(BaseModel):
    project_id: str
    aspect_ratio: str = "9:16"
    target_duration_sec: float = 20.0
    brand_kit: BrandKit
    asset_audit: AssetQualityAudit
    scenes: List[SceneManifest]
    music_vibe: str = "high_energy_modern_electronic"
    estimated_cost_usd: float = 0.0
    verifier_score: Optional[float] = None

