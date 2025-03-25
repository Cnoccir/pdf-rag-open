import { IExecuteFunctions } from 'n8n-core';
import {
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeOperationError,
} from 'n8n-workflow';
import FormData from 'form-data';

export class DoclingExtractor implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Docling Extractor',
		name: 'doclingExtractor',
		group: ['transform'],
		version: 1,
		description: 'Process PDF documents using Docling to extract text, tables, and images',
		defaults: {
			name: 'Docling Extractor',
			color: '#125580',
		},
		inputs: ['main'],
		outputs: ['main'],
		properties: [
			{
				displayName: 'Docling API URL',
				name: 'apiUrl',
				type: 'string',
				default: 'http://localhost:8000/extract',
				required: true,
				description: 'URL of the Docling extraction API',
			},
			{
				displayName: 'PDF Document',
				name: 'binaryPropertyName',
				type: 'string',
				default: 'data',
				required: true,
				description: 'Name of the binary property that contains the PDF file',
			},
			{
				displayName: 'PDF ID',
				name: 'pdfId',
				type: 'string',
				default: '',
				required: false,
				description: 'Optional identifier for the PDF document (will be generated if not provided)',
			},
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				options: [
					{
						displayName: 'Extract Technical Terms',
						name: 'extractTechnicalTerms',
						type: 'boolean',
						default: true,
						description: 'Whether to extract technical terms from the document',
					},
					{
						displayName: 'Extract Procedures',
						name: 'extractProcedures',
						type: 'boolean',
						default: true,
						description: 'Whether to extract procedures and parameters from the document',
					},
					{
						displayName: 'Extract Relationships',
						name: 'extractRelationships',
						type: 'boolean',
						default: true,
						description: 'Whether to extract concept relationships from the document',
					},
					{
						displayName: 'Process Images',
						name: 'processImages',
						type: 'boolean',
						default: true,
						description: 'Whether to process and extract images from the document',
					},
					{
						displayName: 'Process Tables',
						name: 'processTables',
						type: 'boolean',
						default: true,
						description: 'Whether to process and extract tables from the document',
					},
				],
			},
			{
				displayName: 'Output Format',
				name: 'outputFormat',
				type: 'options',
				options: [
					{
						name: 'Combined Object',
						value: 'combined',
						description: 'Return all data in a single JSON object',
					},
					{
						name: 'Separate Items',
						value: 'separate',
						description: 'Return separate items for markdown, tables, images, etc.',
					},
				],
				default: 'combined',
				description: 'Specify how the extracted data should be returned',
			},
			{
				displayName: 'Prepare for Qdrant',
				name: 'prepareForQdrant',
				type: 'boolean',
				default: false,
				description: 'Prepare the extracted data for ingestion into Qdrant',
			},
			{
				displayName: 'Prepare for MongoDB',
				name: 'prepareForMongoDB',
				type: 'boolean',
				default: false,
				description: 'Prepare the extracted data for ingestion into MongoDB',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnItems: INodeExecutionData[] = [];

		// For each item
		for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
			try {
				// Get parameters
				const apiUrl = this.getNodeParameter('apiUrl', itemIndex) as string;
				const binaryPropertyName = this.getNodeParameter('binaryPropertyName', itemIndex) as string;
				const pdfId = this.getNodeParameter('pdfId', itemIndex, '') as string;
				const options = this.getNodeParameter('options', itemIndex, {}) as {
					extractTechnicalTerms?: boolean;
					extractProcedures?: boolean;
					extractRelationships?: boolean;
					processImages?: boolean;
					processTables?: boolean;
				};
				const outputFormat = this.getNodeParameter('outputFormat', itemIndex) as string;
				const prepareForQdrant = this.getNodeParameter('prepareForQdrant', itemIndex, false) as boolean;
				const prepareForMongoDB = this.getNodeParameter('prepareForMongoDB', itemIndex, false) as boolean;

				// Get binary data
				if (items[itemIndex].binary === undefined) {
					throw new NodeOperationError(
						this.getNode(),
						'No binary data exists on item!',
						{ itemIndex },
					);
				}

				const binaryData = items[itemIndex].binary[binaryPropertyName];
				if (binaryData === undefined) {
					throw new NodeOperationError(
						this.getNode(),
						`No binary data property "${binaryPropertyName}" does not exists on item!`,
						{ itemIndex },
					);
				}

				// Create FormData
				const formData = new FormData();
				
				// Add PDF ID if provided
				if (pdfId) {
					formData.append('pdf_id', pdfId);
				}
				
				// Add processing options
				if (options.extractTechnicalTerms !== undefined) {
					formData.append('extract_technical_terms', options.extractTechnicalTerms.toString());
				}
				if (options.extractProcedures !== undefined) {
					formData.append('extract_procedures', options.extractProcedures.toString());
				}
				if (options.extractRelationships !== undefined) {
					formData.append('extract_relationships', options.extractRelationships.toString());
				}
				if (options.processImages !== undefined) {
					formData.append('process_images', options.processImages.toString());
				}
				if (options.processTables !== undefined) {
					formData.append('process_tables', options.processTables.toString());
				}

				// Add file
				const buffer = Buffer.from(binaryData.data, 'base64');
				formData.append('file', buffer, {
					filename: binaryData.fileName || 'document.pdf',
					contentType: binaryData.mimeType,
				});

				// Make request to API
				const requestOptions = {
					method: 'POST',
					body: formData,
					headers: {
						...formData.getHeaders(),
					},
				};

				const response = await this.helpers.request(apiUrl, requestOptions);
				const responseData = typeof response === 'string' ? JSON.parse(response) : response;

				// Process the response based on output format
				if (outputFormat === 'combined') {
					// Create a single item with all data
					const newItem: INodeExecutionData = {
						json: responseData,
						binary: {},
					};

					// Prepare for Qdrant if requested
					if (prepareForQdrant) {
						// Generate vector-ready items with embeddings structure
						newItem.json.qdrantPayload = this.prepareForQdrantStorage(responseData);
					}

					// Prepare for MongoDB if requested
					if (prepareForMongoDB) {
						// Generate MongoDB-ready items with hierarchical document structure
						newItem.json.mongoDBPayload = this.prepareForMongoDBStorage(responseData);
					}

					returnItems.push(newItem);
				} else {
					// Create separate items for different types of extracted content
					
					// Main Markdown item
					returnItems.push({
						json: {
							pdf_id: responseData.pdf_id,
							content_type: 'markdown',
							content: responseData.markdown,
							metadata: responseData.metadata,
						},
					});

					// Table items
					for (const [index, table] of responseData.tables.entries()) {
						returnItems.push({
							json: {
								pdf_id: responseData.pdf_id,
								content_type: 'table',
								index,
								...table,
							},
						});
					}

					// Image items
					for (const [index, imagePath] of responseData.images.entries()) {
						returnItems.push({
							json: {
								pdf_id: responseData.pdf_id,
								content_type: 'image',
								index,
								path: imagePath,
							},
						});
					}

					// Technical terms item
					returnItems.push({
						json: {
							pdf_id: responseData.pdf_id,
							content_type: 'technical_terms',
							terms: responseData.technical_terms,
						},
					});

					// Procedures items
					for (const [index, procedure] of responseData.procedures.entries()) {
						returnItems.push({
							json: {
								pdf_id: responseData.pdf_id,
								content_type: 'procedure',
								index,
								...procedure,
							},
						});
					}

					// Parameters items
					for (const [index, parameter] of responseData.parameters.entries()) {
						returnItems.push({
							json: {
								pdf_id: responseData.pdf_id,
								content_type: 'parameter',
								index,
								...parameter,
							},
						});
					}

					// Relationships item
					returnItems.push({
						json: {
							pdf_id: responseData.pdf_id,
							content_type: 'relationships',
							relationships: responseData.concept_relationships,
						},
					});
				}
			} catch (error) {
				if (this.continueOnFail()) {
					returnItems.push({
						json: {
							error: error.message,
						},
					});
					continue;
				}
				throw error;
			}
		}

		return [returnItems];
	}

	/**
	 * Prepare extracted data for Qdrant storage
	 */
	private prepareForQdrantStorage(data: any): any {
		// In a real implementation, this would prepare the data for Qdrant
		// with proper vector structure, metadata, and payload fields
		
		const qdrantPayload = {
			pdf_id: data.pdf_id,
			markdown_chunks: this.chunkText(data.markdown, 500),
			tables: data.tables.map((table: any) => ({
				table_id: table.table_id,
				caption: table.caption,
				metadata: {
					page: table.page,
					pdf_id: data.pdf_id,
				},
			})),
			technical_terms: data.technical_terms.map((term: string) => ({
				term,
				pdf_id: data.pdf_id,
			})),
			procedures: data.procedures.map((proc: any) => ({
				id: proc.id,
				title: proc.title,
				pdf_id: data.pdf_id,
			})),
		};
		
		return qdrantPayload;
	}

	/**
	 * Prepare extracted data for MongoDB storage
	 */
	private prepareForMongoDBStorage(data: any): any {
		// In a real implementation, this would prepare the data for MongoDB
		// with proper document structure
		
		const mongoDBPayload = {
			pdf_id: data.pdf_id,
			content: {
				markdown: data.markdown,
				technical_terms: data.technical_terms,
				processing_time: data.processing_time,
			},
			tables: data.tables,
			images: data.images.map((path: string) => ({ path })),
			procedures: data.procedures,
			parameters: data.parameters,
			relationships: data.concept_relationships,
			metadata: {
				...data.metadata,
				extracted_at: new Date().toISOString(),
			},
		};
		
		return mongoDBPayload;
	}

	/**
	 * Simple text chunking helper
	 */
	private chunkText(text: string, chunkSize: number): any[] {
		const chunks = [];
		const sentences = text.split('. ');
		let currentChunk = '';
		
		for (const sentence of sentences) {
			if ((currentChunk + sentence).length > chunkSize && currentChunk) {
				chunks.push({
					content: currentChunk,
					size: currentChunk.length,
				});
				currentChunk = sentence + '. ';
			} else {
				currentChunk += sentence + '. ';
			}
		}
		
		if (currentChunk) {
			chunks.push({
				content: currentChunk,
				size: currentChunk.length,
			});
		}
		
		return chunks;
	}
}
