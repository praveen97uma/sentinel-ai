/*
 * Copyright (c) 2025 Original Author(s), PhonePe India Pvt. Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.phonepe.sentinelai.core.agent;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

/**
 * Base interface for multimodal content parts in agent inputs.
 * Supports text, images, and other modalities that can be sent to LLMs.
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(name = "IMAGE", value = ImageFileAttachment.class)
})
public interface Attachment {
    /**
     * Enum representing different types of content
     */
    enum AttachmentType {
        IMAGE,
        AUDIO,
    }

    interface Visitor<T> {
        T visit(ImageFileAttachment image);
    }

    <T> T accept(Visitor<T> visitor);

    /**
     * Gets the type of this content part
     *
     * @return The content type
     */
    AttachmentType getType();
}
