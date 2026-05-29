FROM node:20-alpine AS base

FROM base AS deps
WORKDIR /app
COPY package.json package-lock.json ./
COPY apps/web/package.json apps/web/
COPY packages/splattie-widget/package.json packages/splattie-widget/
RUN npm ci

FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# NEXT_PUBLIC_* are baked into the client bundle at build time. Default to the
# public API host so the analytics beacon never falls back to localhost.
ARG NEXT_PUBLIC_API_URL=https://api.splattie.app
ARG NEXT_PUBLIC_SELF_HOST=false
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_SELF_HOST=$NEXT_PUBLIC_SELF_HOST
RUN npm run build -w apps/web

FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production
RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs
COPY --from=builder /app/apps/web/public ./apps/web/public
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/static ./apps/web/.next/static
USER nextjs
EXPOSE 4001
ENV PORT=4001
ENV HOSTNAME="0.0.0.0"
CMD ["node", "apps/web/server.js"]
